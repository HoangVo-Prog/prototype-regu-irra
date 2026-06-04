import logging
import time
import torch
import torch.distributed as dist
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, get_world_size, synchronize
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _prototype_requested(args):
    return bool(getattr(args, "prototype", False) or getattr(args, "use_loss_id", False))


def _prototype_branch(model):
    return getattr(_unwrap_model(model), "prototype_branch", None)


def _prototype_ready(model):
    branch = _prototype_branch(model)
    return branch is not None and branch.is_ready()


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _meter_scalar(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return value


def _build_prototype_init_loader(args, train_loader):
    source_dataset = getattr(train_loader.dataset, "dataset", None)
    if source_dataset is None:
        raise RuntimeError("Cannot build prototype init loader from this train dataset")

    transforms = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    init_set = ImageTextDataset(
        source_dataset,
        transform=transforms,
        text_length=args.text_length,
    )
    return DataLoader(
        init_set,
        batch_size=getattr(args, "test_batch_size", args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )


@torch.no_grad()
def _collect_prototype_features(model, init_loader, device):
    model = _unwrap_model(model)
    was_training = model.training
    model.eval()

    image_features, text_features, pids = [], [], []
    try:
        for batch in init_loader:
            batch = _move_batch_to_device(batch, device)
            image_feat, text_feat = model.extract_prototype_features(batch)
            image_features.append(image_feat.detach().cpu())
            text_features.append(text_feat.detach().cpu())
            pids.append(batch["pids"].detach().cpu())
    finally:
        if was_training:
            model.train()

    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0), torch.cat(pids, dim=0)


@torch.no_grad()
def _project_prototype_features(branch, image_features, text_features, device, chunk_size):
    projected_images, projected_texts = [], []
    for start in range(0, image_features.shape[0], chunk_size):
        stop = start + chunk_size
        image_chunk = image_features[start:stop].to(device)
        text_chunk = text_features[start:stop].to(device)
        image_projected, text_projected = branch.project_for_memory(image_chunk, text_chunk)
        projected_images.append(image_projected.detach())
        projected_texts.append(text_projected.detach())
    return torch.cat(projected_images, dim=0), torch.cat(projected_texts, dim=0)


@torch.no_grad()
def _broadcast_prototype_branch(branch):
    if get_world_size() <= 1:
        return
    for tensor in branch.state_dict().values():
        dist.broadcast(tensor, src=0)


def maybe_initialize_prototypes(args, model, train_loader, device, logger):
    if not _prototype_requested(args) or _prototype_ready(model):
        return

    branch = _prototype_branch(model)
    if branch is None:
        return

    if get_rank() == 0:
        logger.info("Initializing prototype memory from the full train set")
        init_loader = _build_prototype_init_loader(args, train_loader)
        image_features, text_features, pids = _collect_prototype_features(model, init_loader, device)
        branch.initialize_projector_from_features(image_features, text_features)
        image_projected, text_projected = _project_prototype_features(
            branch,
            image_features,
            text_features,
            device,
            chunk_size=getattr(args, "test_batch_size", args.batch_size),
        )
        branch.initialize_projected(image_projected, text_projected, pids.to(device))
        logger.info(
            "Prototype memory initialized with {} samples and {} identities".format(
                pids.numel(), branch.memory.num_classes
            )
        )

    _broadcast_prototype_branch(branch)
    synchronize()


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "proto_id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    prototype_diagnostic_state = {}

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        if epoch > getattr(args, "prototype_warmup_epochs", 5):
            maybe_initialize_prototypes(args, model, train_loader, device, logger)
            model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            ret = model(batch)
            proto_diag = ret.pop("_proto_diag", None)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            arguments["iteration"] += 1

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(_meter_scalar(ret.get('sdm_loss', 0)), batch_size)
            meters['itc_loss'].update(_meter_scalar(ret.get('itc_loss', 0)), batch_size)
            meters['id_loss'].update(_meter_scalar(ret.get('id_loss', 0)), batch_size)
            meters['proto_id_loss'].update(_meter_scalar(ret.get('proto_id_loss', 0)), batch_size)
            meters['mlm_loss'].update(_meter_scalar(ret.get('mlm_loss', 0)), batch_size)

            meters['img_acc'].update(_meter_scalar(ret.get('img_acc', 0)), batch_size)
            meters['txt_acc'].update(_meter_scalar(ret.get('txt_acc', 0)), batch_size)
            meters['mlm_acc'].update(_meter_scalar(ret.get('mlm_acc', 0)), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                proto_metrics = {}
                branch = _prototype_branch(model)
                if branch is not None and proto_diag is not None:
                    proto_metrics = branch.compute_diagnostics(
                        proto_diag, state=prototype_diagnostic_state
                    )
                    for metric_key, metric_value in proto_metrics.items():
                        tb_writer.add_scalar(metric_key, metric_value, arguments["iteration"])

                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                for k, v in proto_metrics.items():
                    info_str += f", {k.split('/')[-1]}: {v:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        if meters["proto_id_loss"].avg > 0:
            tb_writer.add_scalar(
                "train/weighted_loss/proto_id_loss", meters["proto_id_loss"].avg, epoch
            )


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
