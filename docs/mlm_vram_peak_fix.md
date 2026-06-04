# MLM VRAM Peak Fix Guide

This note documents a common VRAM problem in masked language modeling branches
and how to fix it safely in another PyTorch training codebase.

## Problem Summary

Training can show a very high VRAM peak at the beginning of training, even when
the mini-batch size is small. In this repo the problem became visible when
`--MLM` was enabled.

Two patterns caused the spike:

1. The MLM branch computed vocabulary logits for every token position:

   ```text
   [batch_size, text_length, vocab_size]
   ```

   With CLIP-style settings this becomes:

   ```text
   [B, 77, 49408]
   ```

   Most positions are ignored by `CrossEntropyLoss(ignore_index=0)`, but the
   full logits are already allocated before the loss ignores them.

2. The training meters stored loss tensors directly. If an `AverageMeter` keeps
   a tensor that still has autograd history, it can retain the whole computation
   graph after `backward()`. With MLM enabled, that graph includes the extra
   text encoder pass, cross-modal transformer, and large vocabulary logits.

## How To Recognize This Bug

Look for MLM code shaped like this:

```python
hidden = model(...)
logits = mlm_head(hidden)                  # [B, L, V]
scores = logits.float().reshape(-1, vocab)
labels = batch["mlm_labels"].reshape(-1)
loss = F.cross_entropy(scores, labels, ignore_index=0)
```

This computes logits for all sequence positions. The `ignore_index` only changes
the loss contribution. It does not avoid the allocation.

Also look for metric accumulation like this:

```python
meters["mlm_loss"].update(ret["mlm_loss"], batch_size)
meters["proto_id_loss"].update(ret["proto_id_loss"], batch_size)
meters["img_acc"].update(ret["img_acc"], batch_size)
```

If `ret["..."]` is a tensor, the meter may keep the autograd graph alive.

## Memory Scale

For `text_length = 77` and `vocab_size = 49408`, the raw vocabulary logits are:

```text
B=64:  64 * 77 * 49408 = 243,482,624 logits
B=256: 256 * 77 * 49408 = 973,930,496 logits
```

Approximate memory for just the MLM logits and loss temporaries:

| Batch | fp16 logits | fp32 cast | loss/temp tensors | rough MLM peak |
| ---: | ---: | ---: | ---: | ---: |
| 64 | ~0.45 GiB | ~0.91 GiB | ~0.91 GiB+ | ~2.3-3.2 GiB |
| 256 | ~1.81 GiB | ~3.63 GiB | ~3.63 GiB+ | ~9.1-12.7 GiB |

This is only the MLM vocab projection and loss area. The full training peak also
includes model weights, activations, gradients, optimizer state, CUDA workspaces,
and any additional branches.

## Fix 1: Project Only Masked Positions

The key idea is to run the contextual encoder and any sequence-aware fusion as
usual, then select only the masked positions before the vocabulary projection.

Before:

```python
x = cross_former(mlm_feats, image_feats, image_feats)
x = mlm_head(x)

scores = x.float().reshape(-1, vocab_size)
mlm_labels = batch["mlm_labels"].reshape(-1)
mlm_loss = compute_mlm(scores, mlm_labels)
```

After:

```python
mlm_labels = batch["mlm_labels"].reshape(-1)

x = cross_former(mlm_feats, image_feats, image_feats)
x = x.reshape(-1, x.shape[-1])
masked = mlm_labels != 0

x = mlm_head(x[masked])
scores = x.float().reshape(-1, vocab_size)
mlm_labels = mlm_labels[masked]
mlm_loss = compute_mlm(scores, mlm_labels)
```

Accuracy should be computed on the same filtered tensors:

```python
pred = scores.max(1)[1]
mlm_acc = (pred == mlm_labels).float().mean()
```

### Robust Empty-Mask Guard

Some datasets do not guarantee that every batch has a masked token. In that
case, add an empty-mask guard:

```python
if masked.any():
    scores = mlm_head(x[masked]).float().reshape(-1, vocab_size)
    labels = mlm_labels[masked]
    mlm_loss = compute_mlm(scores, labels)
    mlm_acc = (scores.argmax(dim=1) == labels).float().mean()
else:
    mlm_loss = x.sum() * 0.0
    mlm_acc = x.new_zeros(())
```

This repo's dataset forces at least one masked token per sample, so the guard is
not required here. It is useful for a generic fix.

## When This Fix Is Safe

This fix is safe when the operations after masking are position-wise, such as:

```text
Linear
GELU/ReLU
LayerNorm over feature dim
Final vocab Linear
```

It is also safe when all sequence-aware operations have already happened before
masking, such as:

```text
text transformer
cross attention
cross-modal transformer
```

If an MLM head itself contains sequence-aware operations, apply those operations
before masking, then mask immediately before the expensive final vocabulary
projection.

## Fix 2: Detach Metrics Before Storing

Training meters should store Python numbers, not autograd tensors.

Add a helper:

```python
def meter_scalar(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return value
```

Then use it for every scalar metric from the model output:

```python
meters["sdm_loss"].update(meter_scalar(ret.get("sdm_loss", 0)), batch_size)
meters["id_loss"].update(meter_scalar(ret.get("id_loss", 0)), batch_size)
meters["mlm_loss"].update(meter_scalar(ret.get("mlm_loss", 0)), batch_size)
meters["proto_id_loss"].update(meter_scalar(ret.get("proto_id_loss", 0)), batch_size)
meters["img_acc"].update(meter_scalar(ret.get("img_acc", 0)), batch_size)
meters["txt_acc"].update(meter_scalar(ret.get("txt_acc", 0)), batch_size)
meters["mlm_acc"].update(meter_scalar(ret.get("mlm_acc", 0)), 1)
```

This prevents `AverageMeter.sum`, `AverageMeter.val`, or `AverageMeter.avg` from
holding graph-connected tensors.

## Codex Checklist For Another Codebase

When asking Codex to fix this elsewhere, tell it to do the following:

1. Search for MLM or language modeling branches:

   ```text
   rg -n "mlm|masked|vocab|ignore_index|CrossEntropyLoss|lm_head|mlm_head"
   ```

2. Find logits shaped like `[B, L, V]` or reshaped as `[-1, vocab_size]`.

3. Find labels with an ignore value, commonly `0`, `-100`, or `ignore_index`.

4. Move the label flattening before the vocab projection.

5. Flatten hidden states to `[B * L, D]`.

6. Select only supervised positions:

   ```python
   supervised = labels != ignore_index
   hidden = hidden[supervised]
   labels = labels[supervised]
   ```

7. Run the expensive vocab projection only on `hidden`.

8. Compute loss and accuracy only on the filtered labels.

9. Search training meters/loggers:

   ```text
   rg -n "AverageMeter|meter|\\.update\\(|add_scalar|ret\\.get"
   ```

10. Ensure meters store detached Python scalars, not tensors.

11. Run syntax checks and available unit tests.

12. If possible, compare a short run with:

   ```python
   torch.cuda.reset_peak_memory_stats()
   ...
   print(torch.cuda.max_memory_allocated() / 1024**3)
   ```

## Common Mistakes

- Relying on `ignore_index` to save memory. It does not.
- Masking before sequence-aware encoder layers. That changes model behavior.
- Keeping tensor losses in metric meters.
- Calling `.item()` before `backward()` on the actual loss. Only detach metric
  copies, not the loss used for optimization.
- Filtering logits but forgetting to filter labels the same way.
- Computing accuracy with indices from the old unfiltered label tensor.

## Quick Prompt For Codex

Use this prompt when applying the fix to another repo:

```text
Find the MLM/language-modeling branch that computes full [batch, seq_len, vocab]
logits before CrossEntropyLoss with ignore_index. Patch it so the model runs all
contextual/sequence-aware layers first, then flattens hidden states and labels,
filters to labels != ignore_index, and applies the vocab projection only to
those supervised positions. Also inspect training meters/loggers and ensure they
store detached Python scalars rather than autograd tensors. Keep behavior
equivalent for supervised positions and run syntax/tests.
```
