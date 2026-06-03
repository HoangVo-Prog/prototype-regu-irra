from types import SimpleNamespace
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.prototype import PrototypeBranch, _direction_identity_loss


def _args(**overrides):
    values = {
        "prototype_dim": 4,
        "prototype_projector": "default",
        "prototype_residual_scale": 0.1,
        "prototype_per_id": 2,
        "prototype_momentum": 0.5,
        "prototype_kmeans_iters": 2,
        "prototype_tau": 0.1,
        "prototype_hard_k": 2,
        "no_pbt": False,
        "seed": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PrototypeBranchTest(unittest.TestCase):
    def test_cold_branch_projects_and_returns_zero_loss(self):
        branch = PrototypeBranch(_args(), num_classes=3, image_dim=6, text_dim=5)
        images = torch.randn(4, 6, requires_grad=True)
        texts = torch.randn(4, 5, requires_grad=True)
        pids = torch.tensor([0, 1, 2, 1])

        projected_images, projected_texts = branch.project_for_memory(images, texts)
        self.assertEqual(projected_images.shape, (4, 4))
        self.assertEqual(projected_texts.shape, (4, 4))
        self.assertEqual(projected_images.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(projected_images.norm(dim=1), torch.ones(4), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(projected_texts.norm(dim=1), torch.ones(4), atol=1e-5)
        )

        ret = branch(images, texts, pids, use_loss_id=True)
        self.assertEqual(ret["proto_id_loss"].item(), 0.0)
        ret["proto_id_loss"].backward()
        self.assertIsNotNone(images.grad)

    def test_initialize_assign_update_and_state_dict_roundtrip(self):
        branch = PrototypeBranch(
            _args(prototype_projector="identity"), num_classes=3, image_dim=4, text_dim=4
        )
        pids = torch.tensor([0, 0, 1, 1, 2, 2])
        images = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.9, 0.1],
            ]
        )
        texts = images.roll(shifts=1, dims=1)

        branch.initialize_projected(images, texts, pids)
        self.assertTrue(branch.is_ready())
        self.assertEqual(branch.memory.image_prototypes.shape, (6, 4))
        self.assertTrue(
            torch.equal(branch.memory.proto_pids, torch.tensor([0, 0, 1, 1, 2, 2]))
        )

        assignments = branch.memory.assign_identity(
            images, pids, branch.memory.image_prototypes
        )
        self.assertTrue(torch.all(assignments >= pids * 2))
        self.assertTrue(torch.all(assignments < (pids + 1) * 2))

        before = branch.memory.image_prototypes.clone()
        warm_images = (images + 0.05 * torch.randn_like(images)).requires_grad_()
        warm_texts = (texts + 0.05 * torch.randn_like(texts)).requires_grad_()
        ret = branch(warm_images, warm_texts, pids, use_loss_id=True)
        self.assertTrue(torch.isfinite(ret["proto_id_loss"]))
        ret["proto_id_loss"].backward()
        self.assertIsNotNone(warm_images.grad)
        self.assertFalse(torch.allclose(before, branch.memory.image_prototypes))

        restored = PrototypeBranch(
            _args(prototype_projector="identity"), num_classes=3, image_dim=4, text_dim=4
        )
        restored.load_state_dict(branch.state_dict())
        self.assertTrue(restored.is_ready())
        self.assertTrue(
            torch.allclose(restored.memory.image_prototypes, branch.memory.image_prototypes)
        )

    def test_hard_negatives_exclude_same_identity_prototypes(self):
        features = torch.tensor([[1.0, 0.0]])
        prototypes = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        proto_pids = torch.tensor([0, 0, 1])
        pids = torch.tensor([0])

        loss = _direction_identity_loss(
            features, prototypes, proto_pids, pids, tau=0.1, hard_k=3
        )
        self.assertLess(loss.item(), 1e-4)

    def test_projector_constraints_raise_clear_errors(self):
        with self.assertRaises(ValueError):
            PrototypeBranch(
                _args(prototype_projector="identity", prototype_dim=4),
                num_classes=2,
                image_dim=5,
                text_dim=4,
            )

        with self.assertRaises(ValueError):
            PrototypeBranch(
                _args(prototype_projector="shared", prototype_dim=4),
                num_classes=2,
                image_dim=4,
                text_dim=5,
            )


if __name__ == "__main__":
    unittest.main()
