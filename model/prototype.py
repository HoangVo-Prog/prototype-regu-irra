import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-12


def _normalize(features, dim=-1):
    return F.normalize(features.float(), p=2, dim=dim, eps=_EPS)


class IdentityProjector(nn.Module):
    def forward(self, features):
        return features.float()


class ResidualIdentityProjector(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.residual = nn.Linear(dim, dim)
        self.scale = scale
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        features = features.float()
        return features + self.scale * self.residual(features)


def _linear(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.normal_(layer.weight, std=0.02)
    nn.init.zeros_(layer.bias)
    return layer


def _default_projector(in_dim, out_dim):
    return nn.Sequential(_linear(in_dim, out_dim), nn.LayerNorm(out_dim))


def _orthogonal_projector(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _copy_pca_components(linear, features):
    features = features.float().cpu()
    if linear.out_features > features.shape[1]:
        raise ValueError(
            "prototype_dim must be <= feature dimension for PCA initialized projectors"
        )
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:linear.out_features].to(linear.weight.device, dtype=linear.weight.dtype)
    linear.weight.copy_(components)
    if linear.bias is not None:
        linear.bias.zero_()


def _spherical_kmeans(features, k, num_iters, seed):
    features = _normalize(features)
    n = features.shape[0]
    if n <= k:
        repeats = int(math.ceil(float(k) / float(n)))
        return _normalize(features.repeat((repeats, 1))[:k])

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).to(features.device)
    centroids = features[perm[:k]].clone()

    for _ in range(num_iters):
        sims = features @ centroids.t()
        assignments = sims.argmax(dim=1)
        next_centroids = centroids.clone()
        for slot in range(k):
            mask = assignments == slot
            if mask.any():
                next_centroids[slot] = _normalize(features[mask].mean(dim=0), dim=0)
        centroids = _normalize(next_centroids)

    return centroids


def identity_spherical_kmeans(features, pids, num_classes, prototypes_per_id, num_iters, seed):
    features = _normalize(features)
    pids = pids.long()
    banks = []
    for pid in range(num_classes):
        mask = pids == pid
        if not mask.any():
            raise ValueError(
                "Prototype initialization did not observe train identity {}".format(pid)
            )
        banks.append(
            _spherical_kmeans(
                features[mask],
                prototypes_per_id,
                num_iters=num_iters,
                seed=int(seed) + pid,
            )
        )
    return torch.cat(banks, dim=0)


def _direction_identity_loss(features, prototypes, proto_pids, pids, tau, hard_k):
    features = _normalize(features)
    prototypes = _normalize(prototypes)
    pids = pids.long()

    logits = features @ prototypes.t()
    logits = logits / tau

    same_id = proto_pids.view(1, -1).eq(pids.view(-1, 1))
    pos_lse = logits.masked_fill(~same_id, float("-inf")).logsumexp(dim=1)

    neg_logits = logits.masked_fill(same_id, float("-inf"))
    num_negatives = neg_logits.shape[1] - int(same_id[0].sum().item())
    if num_negatives > 0 and int(hard_k) > 0:
        k = min(int(hard_k), num_negatives)
        neg_lse = neg_logits.topk(k, dim=1).values.logsumexp(dim=1)
    else:
        neg_lse = torch.full_like(pos_lse, float("-inf"))

    return F.softplus(neg_lse - pos_lse).mean()


def symmetric_identity_proxy_loss(image_features, text_features, pids, memory, tau, hard_k, no_pbt):
    if no_pbt:
        image_bank = memory.text_prototypes
        text_bank = memory.image_prototypes
    else:
        image_bank = memory.text_to_image
        text_bank = memory.image_to_text

    image_loss = _direction_identity_loss(
        image_features, image_bank, memory.proto_pids, pids, tau, hard_k
    )
    text_loss = _direction_identity_loss(
        text_features, text_bank, memory.proto_pids, pids, tau, hard_k
    )
    return 0.5 * (image_loss + text_loss)


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, prototypes_per_id, dim, momentum):
        super().__init__()
        self.num_classes = int(num_classes)
        self.prototypes_per_id = int(prototypes_per_id)
        self.dim = int(dim)
        self.momentum = float(momentum)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive for prototype memory")
        if self.prototypes_per_id <= 0:
            raise ValueError("prototype_per_id must be positive")
        if self.dim <= 0:
            raise ValueError("prototype_dim must be positive")
        total = self.num_classes * self.prototypes_per_id

        self.register_buffer("image_prototypes", torch.zeros(total, self.dim))
        self.register_buffer("text_prototypes", torch.zeros(total, self.dim))
        self.register_buffer("text_to_image", torch.zeros(total, self.dim))
        self.register_buffer("image_to_text", torch.zeros(total, self.dim))
        self.register_buffer(
            "proto_pids",
            torch.arange(self.num_classes, dtype=torch.long).repeat_interleave(
                self.prototypes_per_id
            ),
        )
        self.register_buffer("initialized", torch.zeros((), dtype=torch.uint8))

    def is_ready(self):
        return bool(self.initialized.item())

    def _validate_pids(self, pids):
        if pids.numel() == 0:
            raise ValueError("Prototype initialization received no samples")
        if int(pids.min().item()) < 0 or int(pids.max().item()) >= self.num_classes:
            raise ValueError("Prototype pids must be contiguous train ids in [0, num_classes)")
        unique = torch.unique(pids.detach().cpu()).sort().values
        expected = torch.arange(self.num_classes, dtype=unique.dtype)
        if unique.numel() != expected.numel() or not torch.equal(unique, expected):
            raise ValueError("Prototype initialization must see every contiguous train identity")

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids, num_iters, seed):
        image_features = _normalize(image_features)
        text_features = _normalize(text_features)
        pids = pids.to(image_features.device).long()
        self._validate_pids(pids)

        self.image_prototypes.copy_(
            identity_spherical_kmeans(
                image_features,
                pids,
                self.num_classes,
                self.prototypes_per_id,
                num_iters,
                int(seed) + 1000,
            )
        )
        self.text_prototypes.copy_(
            identity_spherical_kmeans(
                text_features,
                pids,
                self.num_classes,
                self.prototypes_per_id,
                num_iters,
                int(seed) + 1001,
            )
        )
        self.rebuild_translated_banks(image_features, text_features, pids)
        self.initialized.fill_(1)

    def assign_identity(self, features, pids, bank):
        features = _normalize(features)
        pids = pids.to(features.device).long()
        bank = _normalize(bank).view(self.num_classes, self.prototypes_per_id, self.dim)
        local_bank = bank[pids]
        sims = torch.bmm(local_bank, features.unsqueeze(-1)).squeeze(-1)
        local_idx = sims.argmax(dim=1)
        return pids * self.prototypes_per_id + local_idx

    @torch.no_grad()
    def rebuild_translated_banks(self, image_features, text_features, pids):
        image_features = _normalize(image_features)
        text_features = _normalize(text_features)
        pids = pids.to(image_features.device).long()

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)

        self.text_to_image.copy_(self.image_prototypes)
        self.image_to_text.copy_(self.text_prototypes)
        self._replace_group_means(self.text_to_image, text_assign, image_features)
        self._replace_group_means(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def ema_update(self, image_features, text_features, pids):
        image_features = _normalize(image_features)
        text_features = _normalize(text_features)
        pids = pids.to(image_features.device).long()

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)

        self._ema_group_update(self.image_prototypes, image_assign, image_features)
        self._ema_group_update(self.text_prototypes, text_assign, text_features)
        self._ema_group_update(self.text_to_image, text_assign, image_features)
        self._ema_group_update(self.image_to_text, image_assign, text_features)
        return image_assign, text_assign

    @torch.no_grad()
    def _replace_group_means(self, bank, assignments, values):
        for row in assignments.unique():
            idx = int(row.item())
            mean = _normalize(values[assignments == row].mean(dim=0), dim=0)
            bank[idx].copy_(mean)

    @torch.no_grad()
    def _ema_group_update(self, bank, assignments, values):
        for row in assignments.unique():
            idx = int(row.item())
            mean = _normalize(values[assignments == row].mean(dim=0), dim=0)
            updated = (1.0 - self.momentum) * bank[idx] + self.momentum * mean
            bank[idx].copy_(_normalize(updated, dim=0))


class PrototypeBranch(nn.Module):
    def __init__(self, args, num_classes, image_dim, text_dim=None):
        super().__init__()
        text_dim = image_dim if text_dim is None else text_dim
        self.args = args
        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim)
        self.prototype_dim = int(getattr(args, "prototype_dim", image_dim))
        self.mode = getattr(args, "prototype_projector", "default")
        self.no_pbt = bool(getattr(args, "no_pbt", False))
        self.tau = float(getattr(args, "prototype_tau", 0.07))
        self.hard_k = int(getattr(args, "prototype_hard_k", 16))
        if self.prototype_dim <= 0:
            raise ValueError("prototype_dim must be positive")
        if self.tau <= 0:
            raise ValueError("prototype_tau must be positive")
        self.register_buffer("pca_initialized", torch.zeros((), dtype=torch.uint8))

        self.shared_projector = None
        self.image_projector = None
        self.text_projector = None
        self._build_projectors(getattr(args, "prototype_residual_scale", 0.1))

        self.memory = PrototypeMemory(
            num_classes=num_classes,
            prototypes_per_id=getattr(args, "prototype_per_id", 4),
            dim=self.prototype_dim,
            momentum=getattr(args, "prototype_momentum", 0.2),
        )

    def _build_projectors(self, residual_scale):
        if self.mode == "default":
            self.image_projector = _default_projector(self.image_dim, self.prototype_dim)
            self.text_projector = _default_projector(self.text_dim, self.prototype_dim)
        elif self.mode == "identity":
            self._require_equal_dims(self.prototype_dim)
            self.image_projector = IdentityProjector()
            self.text_projector = IdentityProjector()
        elif self.mode == "residual_identity":
            self._require_equal_dims(self.prototype_dim)
            self.image_projector = ResidualIdentityProjector(self.image_dim, residual_scale)
            self.text_projector = ResidualIdentityProjector(self.text_dim, residual_scale)
        elif self.mode == "random_orthogonal":
            self.image_projector = _orthogonal_projector(self.image_dim, self.prototype_dim)
            self.text_projector = _orthogonal_projector(self.text_dim, self.prototype_dim)
        elif self.mode == "pca_init":
            self._require_pca_dims()
            self.image_projector = nn.Linear(self.image_dim, self.prototype_dim)
            self.text_projector = nn.Linear(self.text_dim, self.prototype_dim)
        elif self.mode == "shared":
            self._require_shared_dims()
            self.shared_projector = _default_projector(self.image_dim, self.prototype_dim)
        elif self.mode == "shared_pca_init":
            self._require_shared_dims()
            self._require_pca_dims()
            self.shared_projector = nn.Linear(self.image_dim, self.prototype_dim)
        else:
            raise ValueError("Unsupported prototype_projector: {}".format(self.mode))

    def _require_equal_dims(self, prototype_dim):
        if self.image_dim != self.text_dim or self.image_dim != prototype_dim:
            raise ValueError(
                "{} requires image_dim == text_dim == prototype_dim".format(self.mode)
            )

    def _require_shared_dims(self):
        if self.image_dim != self.text_dim:
            raise ValueError("{} requires image_dim == text_dim".format(self.mode))

    def _require_pca_dims(self):
        if self.prototype_dim > self.image_dim or self.prototype_dim > self.text_dim:
            raise ValueError(
                "{} requires prototype_dim <= image_dim and text_dim".format(self.mode)
            )

    def is_ready(self):
        return self.memory.is_ready()

    def needs_pca_init(self):
        return self.mode in ("pca_init", "shared_pca_init") and not bool(
            self.pca_initialized.item()
        )

    @torch.no_grad()
    def initialize_projector_from_features(self, image_features, text_features):
        if not self.needs_pca_init():
            return
        self.float()
        if self.mode == "pca_init":
            _copy_pca_components(self.image_projector, image_features)
            _copy_pca_components(self.text_projector, text_features)
        elif self.mode == "shared_pca_init":
            _copy_pca_components(self.shared_projector, torch.cat([image_features, text_features], dim=0))
        self.pca_initialized.fill_(1)

    @torch.no_grad()
    def project_for_memory(self, image_features, text_features):
        return self._project(image_features, text_features)

    @torch.no_grad()
    def initialize_projected(self, image_features, text_features, pids):
        self.memory.initialize(
            image_features,
            text_features,
            pids,
            num_iters=getattr(self.args, "prototype_kmeans_iters", 20),
            seed=getattr(self.args, "seed", 1),
        )

    def _project(self, image_features, text_features):
        self.float()
        image_features = image_features.float()
        text_features = text_features.float()

        if self.shared_projector is not None:
            image_features = self.shared_projector(image_features)
            text_features = self.shared_projector(text_features)
        else:
            image_features = self.image_projector(image_features)
            text_features = self.text_projector(text_features)

        return _normalize(image_features), _normalize(text_features)

    def forward(self, image_features, text_features, pids, use_loss_id=True):
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        ret = {}

        if not self.is_ready():
            if use_loss_id:
                ret["proto_id_loss"] = zero
            return ret

        pids = pids.long()
        if use_loss_id:
            ret["proto_id_loss"] = symmetric_identity_proxy_loss(
                image_features,
                text_features,
                pids,
                self.memory,
                tau=self.tau,
                hard_k=self.hard_k,
                no_pbt=self.no_pbt,
            )

        with torch.no_grad():
            image_assign, text_assign = self.memory.ema_update(
                image_features.detach(), text_features.detach(), pids.detach()
            )
            ret["_proto_diag"] = {
                "image_features": image_features.detach(),
                "text_features": text_features.detach(),
                "pids": pids.detach(),
                "image_assign": image_assign.detach(),
                "text_assign": text_assign.detach(),
            }
        return ret

    @torch.no_grad()
    def compute_diagnostics(self, payload, state=None):
        if not self.is_ready() or payload is None:
            return {}

        image_features = payload["image_features"]
        text_features = payload["text_features"]
        pids = payload["pids"].long()
        image_assign = payload.get("image_assign")
        text_assign = payload.get("text_assign")

        if self.no_pbt:
            image_bank = self.memory.text_prototypes
            text_bank = self.memory.image_prototypes
        else:
            image_bank = self.memory.text_to_image
            text_bank = self.memory.image_to_text

        image_margin = self._margin(image_features, pids, image_bank)
        text_margin = self._margin(text_features, pids, text_bank)
        margins = torch.cat([image_margin, text_margin])

        metrics = {
            "train/proto_margin_img_mean": image_margin.mean(),
            "train/proto_margin_txt_mean": text_margin.mean(),
            "train/negative_proto_margin_rate": (margins < 0).float().mean(),
            "train/slot_redundancy": self._slot_redundancy(),
        }

        if image_assign is not None and text_assign is not None:
            dead_rate, effective_slots = self._assignment_metrics(
                image_assign, text_assign, pids
            )
            metrics["train/dead_slot_rate"] = dead_rate
            metrics["train/effective_slots_per_id"] = effective_slots

        indices = payload.get("indices")
        if indices is not None and state is not None:
            flip_rate = self._assignment_flip_rate(indices, image_assign, text_assign, state)
            if flip_rate is not None:
                metrics["train/assignment_flip_rate"] = flip_rate

        finite_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value) and torch.isfinite(value).all():
                finite_metrics[key] = float(value.detach().cpu().item())
        return finite_metrics

    def _margin(self, features, pids, bank):
        sims = _normalize(features) @ _normalize(bank).t()
        same_id = self.memory.proto_pids.view(1, -1).eq(pids.view(-1, 1))
        pos = sims.masked_fill(~same_id, float("-inf")).max(dim=1).values
        neg = sims.masked_fill(same_id, float("-inf")).max(dim=1).values
        no_neg = torch.isinf(neg)
        if no_neg.any():
            neg = neg.masked_fill(no_neg, 0.0)
        return pos - neg

    def _assignment_metrics(self, image_assign, text_assign, pids):
        total = self.memory.num_classes * self.memory.prototypes_per_id
        counts = torch.zeros(total, device=image_assign.device, dtype=torch.float32)
        ones = torch.ones_like(image_assign, dtype=torch.float32)
        counts.scatter_add_(0, image_assign, ones)
        counts.scatter_add_(0, text_assign, torch.ones_like(text_assign, dtype=torch.float32))
        dead_rate = (counts == 0).float().mean()

        effective = []
        for pid in torch.unique(pids):
            start = int(pid.item()) * self.memory.prototypes_per_id
            stop = start + self.memory.prototypes_per_id
            slot_counts = counts[start:stop]
            if slot_counts.sum() > 0:
                probs = slot_counts / slot_counts.sum()
                entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
                effective.append(entropy.exp())
        if effective:
            effective_slots = torch.stack(effective).mean()
        else:
            effective_slots = torch.ones((), device=counts.device)
        return dead_rate, effective_slots

    def _slot_redundancy(self):
        values = []
        for bank in (self.memory.image_prototypes, self.memory.text_prototypes):
            bank = _normalize(bank).view(
                self.memory.num_classes, self.memory.prototypes_per_id, self.memory.dim
            )
            if self.memory.prototypes_per_id <= 1:
                values.append(torch.zeros((), device=bank.device))
                continue
            sims = torch.bmm(bank, bank.transpose(1, 2))
            eye = torch.eye(self.memory.prototypes_per_id, device=bank.device, dtype=torch.bool)
            values.append(sims[:, ~eye].view(self.memory.num_classes, -1).mean())
        return torch.stack(values).mean()

    def _assignment_flip_rate(self, indices, image_assign, text_assign, state):
        if image_assign is None or text_assign is None:
            return None
        flips = []
        indices = indices.detach().cpu().tolist()
        image_assign = image_assign.detach().cpu().tolist()
        text_assign = text_assign.detach().cpu().tolist()
        previous = state.setdefault("prototype_assignments", {})
        for index, image_row, text_row in zip(indices, image_assign, text_assign):
            old = previous.get(index)
            new = (image_row, text_row)
            if old is not None:
                flips.append(float(old != new))
            previous[index] = new
        if not flips:
            return None
        return torch.tensor(sum(flips) / len(flips))
