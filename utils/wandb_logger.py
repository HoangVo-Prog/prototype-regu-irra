import logging


class WandbLogger:
    def __init__(self, run=None):
        self._run = run

    @property
    def enabled(self):
        return self._run is not None

    def log(self, metrics, step=None):
        if not self.enabled or not metrics:
            return
        payload = {}
        for key, value in metrics.items():
            if value is None:
                continue
            payload[key] = value
        if payload:
            self._run.log(payload, step=step)

    def finish(self):
        if self.enabled:
            self._run.finish()


def init_wandb(args, distributed_rank=0, logger=None):
    if logger is None:
        logger = logging.getLogger("IRRA")

    if not getattr(args, "wandb", False):
        return WandbLogger()

    if distributed_rank != 0:
        return WandbLogger()

    try:
        import wandb
    except ImportError:
        logger.warning("W&B requested but the 'wandb' package is not installed; continuing without W&B")
        return WandbLogger()

    tags = getattr(args, "wandb_tags", None)
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

    run_name = getattr(args, "wandb_name", None)
    if not run_name:
        run_name = getattr(args, "run_timestamp", None) or getattr(args, "name", None)

    run = wandb.init(
        project=getattr(args, "wandb_project", None),
        entity=getattr(args, "wandb_entity", None),
        name=run_name,
        tags=tags or None,
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
    )
    logger.info("Initialized W&B run%s", f" '{run.name}'" if getattr(run, "name", None) else "")
    return WandbLogger(run=run)
