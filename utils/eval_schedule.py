def should_evaluate_epoch(epoch, eval_period, eval_after_epoch):
    if eval_after_epoch < 0:
        raise ValueError("--eval_after_epoch must be >= 0")
    return epoch % eval_period == 0 and (eval_after_epoch == 0 or epoch >= eval_after_epoch)
