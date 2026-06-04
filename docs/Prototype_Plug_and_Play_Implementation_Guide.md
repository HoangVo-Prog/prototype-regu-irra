# Prototype Plug-and-Play Implementation Guide

This document guides Codex to implement the current identity-aware prototype
regularizer on a different image-text retrieval codebase.

The guide is intentionally host-agnostic. The source branch currently attaches
the prototype subsystem to one concrete model, but a plug-and-play port should
not copy that model's architecture, evaluator rows, local-feature names, data
factory, or training script details. Copy the prototype algorithm and its
integration contract only.

The target behavior is:

- keep the host model's main retrieval objective unchanged;
- expose one image embedding, one text embedding, and one contiguous train
  identity label per sample;
- initialize prototype memory once from the full train set after warmup;
- add a weighted prototype identity loss during training;
- update prototype memory with no-gradient EMA;
- log prototype losses and diagnostics through the host's existing logging
  system.

## 1. Scope

Implement the fixed-slot identity-aware prototype branch:

- trainable projection heads into a shared prototype space;
- fixed `prototype_per_id` slots per train identity;
- identity-wise spherical K-Means initialization;
- same-side image/text prototype banks;
- translated PBT banks (`text_to_image`, `image_to_text`);
- identity-restricted assignment for initialization and EMA;
- symmetric prototype identity loss with hard wrong-identity prototype
  negatives;
- optional `--no_pbt` ablation that uses raw cross-modal prototype banks instead
  of translated PBT banks.

Do not implement these as part of the first plug-and-play port unless the user
explicitly asks:

- prototype retrieval fusion at validation or test time;
- prototype rank loss;
- adaptive hard-k;
- adaptive prototype budgeting;
- prototype refresh schedules;
- confidence-weighted updates;
- host-specific local attention mining modules.

The current source branch has an optional `score(...)` helper on the prototype
branch, but the active training path only needs the identity regularizer.

Prototype-source utilities to port or re-create:

| Utility | Purpose |
| --- | --- |
| `PrototypeBranch` | Projection heads, cold-memory zero loss, identity loss call, EMA update call, optional score API. |
| `PrototypeMemory` | Prototype buffers, identity assignment, translated bank rebuild, grouped means, EMA updates. |
| Spherical K-Means helper | Per-identity fixed-slot centroid initialization. |
| Identity proxy loss helper | Positive/negative masks, hard prototype negatives, symmetric image/text loss. |
| Prototype init helper | Full-train feature scan, PCA projector initialization, projection, memory initialization. |
| Prototype diagnostics helper | Margins, dead slots, effective slot usage, redundancy, assignment flips, optional host/prototype correlation. |
| W&B/log wrapper hooks | Rank-0/no-op logging and prototype metric key assembly. |
| Optimizer hook | Assign prototype projector parameters to `prototype_lr` or fallback LR. |
| Checkpoint hook | Save and restore prototype projectors and memory using the target checkpoint policy. |

## 2. Minimal Host Contract

For every training mini-batch, the host must provide:

```text
image_features: Tensor[B, D_img]
text_features : Tensor[B, D_txt]
pids          : LongTensor[B]
```

`pids` must be contiguous train identity indices:

```text
0 <= pids[i] < num_train_classes
```

If the host dataset uses arbitrary identity IDs, remap them in the train split
before prototype memory can index rows by identity.

Optional but useful:

```text
indices: LongTensor[B]
```

`indices` are stable train-sample indices used for `assignment_flip_rate`
diagnostics. If the target codebase does not expose them, skip that diagnostic
or log it only when available.

## 3. Feature Selection For A General Backbone

The prototype branch should see embeddings, not host internals.

Create one adapter function in the host model:

```python
def select_prototype_features(self, outputs, batch):
    return image_features, text_features
```

Recommended selection order:

1. Use the same global embeddings the host evaluator uses for retrieval.
2. If the host has multiple retrieval heads, make `--prototype_feature` choose
   among named feature sources.
3. If a local/region feature is experimental or expensive, do not make it the
   default just because it exists.
4. Avoid mixing sources: initialize, train, and diagnose prototypes from the
   same feature source.

Suggested generic config:

```text
--prototype_feature auto|global|local|head_name
```

For a new backbone, define `auto` explicitly. A conservative default is:

```text
auto == host default retrieval embedding
```

If the host has no local feature path, either accept only `global` or make
`auto` and `global` equivalent.

## 4. Config Surface

These options are enough to reproduce the current prototype branch behavior in
a new codebase.

| Option | Purpose |
| --- | --- |
| `--prototype` | Build prototype infrastructure even if the loss is not enabled yet. |
| `--use_loss_id` | Add `proto_id_loss` to the host training objective. Also implies building the branch. |
| `--no_pbt` | Use raw cross-modal prototype banks for identity loss instead of translated PBT banks. |
| `--prototype_feature` | Select the host feature source routed into the branch. |
| `--prototype_projector` | Projection mode: `default`, `identity`, `residual_identity`, `random_orthogonal`, `pca_init`, `shared`, `shared_pca_init`. |
| `--prototype_residual_scale` | Residual scale for `residual_identity`. |
| `--prototype_per_id` | Number of prototype slots per train identity. |
| `--prototype_dim` | Dimension of the shared prototype space. |
| `--prototype_kmeans_iters` | Spherical K-Means iterations during initialization. |
| `--prototype_warmup_epochs` | Initialize when `epoch > prototype_warmup_epochs`. |
| `--prototype_tau` | Temperature for prototype identity loss. |
| `--prototype_hard_k` | Number of hard wrong-identity prototype rows used in the negative term. |
| `--prototype_id_weight` | Weight applied once to `proto_id_loss` before host loss aggregation. |
| `--prototype_momentum` | EMA momentum for prototype memory updates. |
| `--prototype_lr` | Optional absolute LR for prototype projection parameters. |
| `--seed` | Host seed; current branch derives prototype K-Means seeds from it. |

Current branch seed behavior:

```text
image K-Means seed = seed + 1000
text K-Means seed  = seed + 1001
```

For plug-and-play work, keep prototype random operations on a local
`torch.Generator` so prototype initialization does not consume the host's global
RNG stream. If the target project already has a separate seed namespace, adding
`--prototype_seed` is acceptable, but it is not part of the current branch's CLI.

Do not add these options unless implementing a new feature:

```text
--use_loss_rank
--prototype_margin
--prototype_refresh_*
--prototype_score_weight
--prototype_score_weights
```

## 5. Branch Construction

The host model should construct the branch when either prototype flag is set:

```python
self.prototype_enabled = args.prototype or args.use_loss_id

if self.prototype_enabled:
    self.prototype_branch = PrototypeBranch(
        args=args,
        num_classes=num_train_classes,
        image_dim=selected_image_dim,
        text_dim=selected_text_dim,
    )
else:
    self.prototype_branch = None
```

The current source branch assumes image/text input dimensions are equal because
its host feature sources are paired that way. A plug-and-play implementation
should support different dimensions:

```python
image_projector: D_img -> prototype_dim
text_projector : D_txt -> prototype_dim
```

After projection, both modalities live in the same prototype space.

Keep memory as buffers, not parameters. Only projection modules should be
optimized by the host optimizer.

## 6. Projection Modes

Every mode must output float32, L2-normalized vectors in `prototype_dim`.

Default projection:

```python
image_projector = Linear(D_img, prototype_dim) + LayerNorm(prototype_dim)
text_projector  = Linear(D_txt, prototype_dim) + LayerNorm(prototype_dim)
```

Projection rules:

- cast host features to `float32`;
- cast projector modules to `float32` before use if the host converts modules to
  fp16;
- use separate image/text projectors unless the selected mode is shared;
- L2-normalize along the feature dimension;
- initialize and update memory with projected features, not raw host features;
- let gradients flow through projectors via `proto_id_loss`;
- never let optimizer gradients update prototype memory buffers.

Modes:

| Mode | Behavior | Constraints |
| --- | --- | --- |
| `default` | Separate `Linear + LayerNorm` projectors. | Works for different `D_img` and `D_txt`. |
| `identity` | Use raw features after float cast and normalization. | Requires `D_img == D_txt == prototype_dim`. |
| `residual_identity` | `x + scale * Linear(x)` with zero-initialized residual. | Requires `D_img == D_txt == prototype_dim`. |
| `random_orthogonal` | Separate linear projectors with orthogonal weight init and zero bias. | Works when the linear init supports the requested shape. |
| `pca_init` | Separate linear projectors initialized from train-set PCA. | Requires `prototype_dim <= D_img` and `prototype_dim <= D_txt`. |
| `shared` | One shared linear projector for both modalities. | Requires `D_img == D_txt`. |
| `shared_pca_init` | One shared PCA-initialized projector from concatenated image/text train features. | Requires `D_img == D_txt` and `prototype_dim <= D_img`. |

PCA initialization sequence:

1. Collect raw train embeddings during prototype initialization.
2. Center features on CPU.
3. Compute SVD components.
4. Copy components into projector weights.
5. Zero projector bias.
6. Mark the projector as PCA-initialized.
7. Project features through the initialized projectors before K-Means.

## 7. Prototype Memory Layout

For `num_classes` train identities and `K = prototype_per_id`, each bank has:

```text
total_prototypes = num_classes * K
```

Buffers:

```text
image_prototypes: Tensor[total_prototypes, prototype_dim]
text_prototypes : Tensor[total_prototypes, prototype_dim]
text_to_image   : Tensor[total_prototypes, prototype_dim]
image_to_text   : Tensor[total_prototypes, prototype_dim]
proto_pids      : LongTensor[total_prototypes]
initialized     : Bool scalar buffer
```

Identity ownership:

```python
proto_pids = torch.arange(num_classes).repeat_interleave(K)
```

Rows for identity `pid` are contiguous:

```python
start = pid * K
end = (pid + 1) * K
```

This layout is required because assignment reshapes a bank into:

```text
[num_classes, K, prototype_dim]
```

and compares each sample only with the slots owned by its ground-truth identity.

## 8. Full-Train Initialization

Initialize prototype memory once after warmup:

```python
if prototype_requested(args):
    if epoch > args.prototype_warmup_epochs and not prototype_ready(model):
        maybe_initialize_prototypes(...)
```

The initialization source should be the full underlying train dataset, not the
sampled training epoch stream. Identity samplers can drop, repeat, or reorder
examples in ways that are correct for training but wrong for prototype
construction.

Initialization workflow:

1. Put the host model in eval mode.
2. Build a dedicated prototype-init loader over the full train dataset.
3. Use evaluation/no-random-augmentation image transforms.
4. Disable text augmentation or prompt augmentation.
5. Disable random image augmentation if the dataset exposes a flag for it.
6. Iterate once under `torch.no_grad()`.
7. Extract raw image/text embeddings through the same feature adapter used for
   training prototypes.
8. Store raw features and `pids` on CPU.
9. Restore the host model's previous train/eval state.
10. If using PCA projector modes, initialize projectors from raw features.
11. Project raw features in chunks through `project_for_memory(...)`.
12. Validate feature count and identity coverage when possible.
13. Run identity-wise spherical K-Means separately for image and text features.
14. Rebuild translated PBT banks.
15. Mark memory as initialized.

Validation recommended for a portable implementation:

```text
num_seen == len(full_train_dataset)
unique(pids) == arange(num_train_classes)
pids are contiguous and within range
projected feature shapes match [num_seen, prototype_dim]
```

The current low-level K-Means helper can fall back to a global mean if an
identity has no samples. Treat that as a defensive fallback, not a normal
plug-and-play path. A missing train identity usually means the init loader or
label remapping is wrong.

## 9. K-Means Details

Use spherical K-Means over L2-normalized projected features.

Per identity `y`:

```text
image_prototypes[y] = KMeans({projected_image_i | pid_i = y}, K)
text_prototypes[y]  = KMeans({projected_text_i  | pid_i = y}, K)
```

Sparse identities:

- if sample count is less than or equal to `K`, repeat available projected
  features until `K` rows exist;
- if sample count is zero, prefer raising a clear initialization error in the
  port; only keep global-mean fallback if reproducing the exact defensive helper
  behavior.

K-Means implementation rules:

- normalize input features first;
- initialize centroids with a local `torch.Generator`;
- assign by cosine similarity;
- update centroids by normalized cluster means;
- keep the previous centroid for an empty cluster during an iteration;
- chunk similarity computation for large datasets;
- run image and text K-Means with different local seeds.

## 10. Identity-Restricted Assignment

Training memory updates must assign a sample only to prototypes owned by its
ground-truth identity:

```python
def assign_identity(features, pids, bank):
    features = normalize(features)
    local_bank = bank.view(num_classes, K, dim)[pids]
    sims = torch.bmm(local_bank, features.unsqueeze(-1)).squeeze(-1)
    local_idx = sims.argmax(dim=1)
    return pids * K + local_idx
```

Do not use globally nearest prototypes for training updates. Global assignment
can let a wrong identity update a slot because of visual or textual attribute
similarity.

Global assignment is only acceptable for optional inference scoring or analysis.

## 11. Translated PBT Banks

Same-side banks:

```text
image_prototypes
text_prototypes
```

Translated banks:

```text
text_to_image
image_to_text
```

`text_to_image[j]` stores a visual anchor for text prototype row `j`. It is the
mean paired image feature of samples whose text feature assigned to text row
`j`.

`image_to_text[j]` stores a textual anchor for image prototype row `j`. It is
the mean paired text feature of samples whose image feature assigned to image
row `j`.

After K-Means:

```python
text_to_image.copy_(image_prototypes)
image_to_text.copy_(text_prototypes)
```

Then replace rows that received assignments:

```text
text_to_image[j] = normalize(mean image feature for text_assign == j)
image_to_text[j] = normalize(mean text feature  for image_assign == j)
```

Rows with no assignments keep their same-side fallback.

## 12. Online EMA Update

After computing `proto_id_loss`, update memory under `torch.no_grad()` with
detached projected features.

Same-side updates:

```text
image_prototypes[image_assign] <- EMA(mean assigned image features)
text_prototypes[text_assign]   <- EMA(mean assigned text features)
```

Translated updates:

```text
text_to_image[text_assign] <- EMA(mean paired image features)
image_to_text[image_assign] <- EMA(mean paired text features)
```

EMA rule:

```text
bank[row] = normalize((1 - prototype_momentum) * bank[row]
                      + prototype_momentum * batch_mean[row])
```

Rows absent from the current batch are not updated.

Forward order:

```text
project image/text features
if memory is cold:
    return zero proto_id_loss when requested
compute proto_id_loss from current memory
ema_update(detached projected features, pids)
return loss dict
```

Updating memory before loss computation changes the objective and makes stale
memory behavior harder to debug.

## 13. Prototype Identity Loss

The active loss is symmetric identity proxy contrast.

With PBT enabled:

```text
image features compare with text_to_image
text features compare with image_to_text
```

With `--no_pbt`:

```text
image features compare with text_prototypes
text features compare with image_prototypes
```

For one direction:

```python
features = normalize(features)
prototypes = normalize(selected_bank)
logits = features @ prototypes.T
logits = logits / prototype_tau
```

Positive and negative masks:

```text
positive rows: proto_pids == pid
negative rows: proto_pids != pid
```

Positive score:

```text
pos_lse = logsumexp(logits over all same-ID prototype rows)
```

Hard negative score:

```text
hard rows = top prototype_hard_k wrong-ID rows by logit
neg_lse = logsumexp(logits over selected hard rows)
```

Directional loss:

```text
loss = -log(exp(pos_lse) / (exp(pos_lse) + exp(neg_lse)))
     = softplus(neg_lse - pos_lse)
```

Symmetric loss:

```text
proto_id_loss = 0.5 * (image_direction_loss + text_direction_loss)
```

The host model applies:

```python
ret["proto_id_loss"] = proto_id_loss * args.prototype_id_weight
```

Apply `prototype_id_weight` exactly once. The training loop can then keep its
normal behavior of summing all returned tensors whose key contains `"loss"`.

Cold memory behavior:

```python
zero = projected_image_features.sum() * 0.0
ret["proto_id_loss"] = zero
```

The zero should be on the right device and attached to the current graph.

## 14. Host Training Integration

In the host forward pass:

```python
outputs = self.forward_host(batch)
ret = self.compute_host_losses(outputs, batch)

if self.prototype_branch is not None:
    image_feat, text_feat = self.select_prototype_features(outputs, batch)
    proto_ret = self.prototype_branch(
        image_feat,
        text_feat,
        batch["pids"],
        use_loss_id=args.use_loss_id,
    )
    if "proto_id_loss" in proto_ret:
        ret["proto_id_loss"] = proto_ret["proto_id_loss"] * args.prototype_id_weight

return ret
```

For initialization:

```python
@torch.no_grad()
def extract_prototype_features(self, batch):
    outputs = self.forward_features_for_retrieval(batch)
    return self.select_prototype_features(outputs, batch)
```

Keep this path no-grad and eval-mode compatible. It should return raw host
embeddings; the branch handles projection.

## 15. Optimizer Integration

Prototype memory buffers are not optimized. Prototype projectors are optimized.

Add prototype parameters to the host optimizer with:

```text
prototype_lr if provided, otherwise lr * lr_factor
```

If the host optimizer uses parameter groups, match by module ownership instead
of string names when possible:

```python
prototype_params = list(model.prototype_branch.parameters())
```

If the host uses string-name rules, ensure every trainable parameter under the
prototype branch receives the prototype LR.

## 16. Checkpoint Policy

A portable implementation can choose one of two clear policies.

Policy A: normal state dict includes memory buffers.

- simplest for most codebases;
- resuming training restores initialized memory automatically;
- inference scoring, if added later, can load memory with the model.

Policy B: split prototype memory from the main checkpoint.

- mirrors the current branch;
- main checkpoint stores host model and prototype projectors but excludes
  `prototype_branch.memory.*`;
- save an additional prototype bank checkpoint containing memory buffers;
- save branch metadata such as `prototype_dim`, `prototype_per_id`,
  `projector_mode`, `feature_source`, and readiness.

If using Policy B, implement explicit loading for the bank checkpoint. Do not
silently treat zero memory as initialized.

Regardless of policy, checkpoints should preserve:

```text
projection head parameters
prototype memory buffers
proto_pids
initialized
projector mode / feature source metadata
```

## 17. Distributed Training

All ranks must train with consistent initialized memory.

Safe options:

1. Initialize on rank 0, then broadcast all prototype memory buffers and any PCA
   projector weights to other ranks.
2. Initialize on every rank using the same full ordered init loader and same
   local seeds, then synchronize before training continues.

Rank-0 broadcast is less fragile when the target codebase uses stochastic data
transforms, multiple filesystems, or non-identical worker behavior.

After initialization:

```text
synchronize ranks before the next training iteration
```

During normal training, the current branch updates memory locally. If the target
DDP setup requires exact cross-rank memory equality after every step, add a
deliberate all-reduce/broadcast strategy and document it as an extension.

## 18. W&B And Logging

Use the host's existing W&B or logging utilities. Do not create a second logging
path just for prototypes.

The useful generic W&B utility behavior is:

- initialize only on rank 0;
- make disabled/missing W&B a no-op;
- log `config=vars(args)` or the host equivalent;
- call one wrapper such as `wandb_log(metrics, step=global_step)`;
- finish the run in the training `finally` block.

### 18.1 Loss Metrics

Log these when available:

```text
train/weighted_loss/proto_id_loss
train/loss_grad_norm/proto_id_loss
```

`train/weighted_loss/proto_id_loss` should be the already weighted scalar that
participates in the total training loss.

If the host already logs:

```text
train/total_loss
train/weighted_loss
train/lr
train/lr_min
train/lr_max
```

leave those keys unchanged and add prototype metrics into the same namespace.

### 18.2 Prototype Diagnostics

Log these after memory is initialized and values are finite:

```text
train/proto_margin_img_mean
train/proto_margin_txt_mean
train/negative_proto_margin_rate
train/dead_slot_rate
train/effective_slots_per_id
train/slot_redundancy
train/assignment_flip_rate
train/hard_negative_overlap
train/proto_to_host_margin_corr
```

Definitions:

```text
proto_margin_img =
    best same-ID similarity(image_feature, image-direction bank)
  - best wrong-ID similarity(image_feature, image-direction bank)
```

```text
proto_margin_txt =
    best same-ID similarity(text_feature, text-direction bank)
  - best wrong-ID similarity(text_feature, text-direction bank)
```

```text
negative_proto_margin_rate =
    fraction of image/text prototype margins below zero
```

```text
dead_slot_rate =
    fraction of identity-owned slots receiving zero assignments in the current
    diagnostic batch
```

```text
effective_slots_per_id =
    exp(entropy(slot assignment distribution)) averaged over present identities
```

```text
slot_redundancy =
    mean off-diagonal cosine similarity among slots of the same identity
```

```text
assignment_flip_rate =
    fraction of image/text assignments that changed for the same train sample
    since its previous diagnostic observation
```

```text
hard_negative_overlap =
    overlap between host hard-negative identities and prototype hard-negative
    identities, if host hard-negative diagnostics are available
```

```text
proto_to_host_margin_corr =
    correlation between prototype margins and host retrieval margins, if host
    margin diagnostics are available
```

If the target host does not expose host hard negatives or host margins, skip
`hard_negative_overlap` and `proto_to_host_margin_corr` rather than inventing
weak substitutes.

### 18.3 Validation Metrics

Keep the host's validation metrics unchanged. Prototype training does not
require validation-time prototype fusion.

If the host logs validation metrics, it is enough to keep existing keys such as:

```text
val/top1
val/best_top1
val/mAP
val/rSum
```

or whatever the host already uses. Do not add prototype validation rows unless
implementing explicit prototype inference scoring.

## 19. Diagnostic Implementation Notes

The forward pass can return a private diagnostic payload:

```python
ret["_diag"] = {
    "host_image_feats": host_image_features.detach(),
    "host_text_feats": host_text_features.detach(),
    "proto_image_feats": prototype_source_image_features.detach(),
    "proto_text_feats": prototype_source_text_features.detach(),
    "pids": batch["pids"].detach(),
    "indices": batch.get("index"),
}
```

Then the training loop can compute diagnostics every `log_period` instead of
every iteration.

Rules:

- detach diagnostic features;
- project prototype diagnostic features through `project_for_memory(...)` before
  computing prototype metrics;
- compute diagnostics under `torch.no_grad()`;
- log only finite values;
- keep a small state dictionary for assignment flip tracking.

## 20. Optional Prototype Score API

The current branch includes a score helper, but it is not required for the
training regularizer.

If implementing it, keep it separate from the training loss:

```python
@torch.no_grad()
def score(self, text_features, image_features):
    image_features, text_features = self.project_for_memory(image_features, text_features)
    return memory.prototype_score_matrix(text_features, image_features)
```

Score matrix idea:

1. globally assign each text feature to the nearest text prototype;
2. globally assign each image feature to the nearest image prototype;
3. compare gathered translated/same-side prototype rows;
4. return a text-query by image-gallery score matrix.

Do not make validation depend on this score path unless the user explicitly
requests inference fusion.

## 21. Implementation Skeleton

Minimal branch API:

```python
class PrototypeBranch(nn.Module):
    def __init__(self, args, num_classes, image_dim, text_dim=None):
        ...

    def is_ready(self):
        return self.memory.is_ready()

    def needs_pca_init(self):
        ...

    @torch.no_grad()
    def initialize_projector_from_features(self, image_features, text_features):
        ...

    @torch.no_grad()
    def project_for_memory(self, image_features, text_features):
        return self._project(image_features, text_features)

    @torch.no_grad()
    def initialize_projected(self, image_features, text_features, pids):
        self.memory.initialize(...)

    def forward(self, image_features, text_features, pids, use_loss_id=True):
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        if not self.is_ready():
            return {"proto_id_loss": zero} if use_loss_id else {}

        ret = {}
        if use_loss_id:
            ret["proto_id_loss"] = symmetric_identity_proxy_loss(...)

        self.memory.ema_update(
            image_features.detach(),
            text_features.detach(),
            pids.detach(),
        )
        return ret
```

Minimal memory API:

```python
class PrototypeMemory(nn.Module):
    def is_ready(self): ...
    def initialize(self, image_features, text_features, pids, num_iters, seed): ...
    def assign_identity(self, features, pids, bank): ...
    def ema_update(self, image_features, text_features, pids): ...
```

Minimal training-loop hook:

```python
if args.prototype or args.use_loss_id:
    if epoch > args.prototype_warmup_epochs and not prototype_ready(model):
        maybe_initialize_prototypes(model, train_loader, args, device, logger)
```

## 22. Verification Checklist

Construction:

```text
prototype_branch is None when prototype flags are false
prototype_branch exists when --prototype or --use_loss_id is true
optimizer includes projector parameters
memory banks are buffers, not parameters
```

Projection:

```text
image projection shape == [B, prototype_dim]
text projection shape == [B, prototype_dim]
projected features are float32
projected features are L2-normalized
projector mode constraints raise clear errors
```

Initialization:

```text
init loader scans the full train dataset
augmentation is disabled for init
all train identities are present
banks have shape [num_classes * prototype_per_id, prototype_dim]
proto_pids is contiguous repeat_interleave layout
initialized == true after bank construction
```

Assignment:

```text
assign_identity always returns rows inside [pid*K, (pid+1)*K)
EMA uses identity-restricted assignment
global assignment is not used for training updates
```

Loss:

```text
cold memory returns zero proto_id_loss
warm memory returns finite proto_id_loss
hard negatives never include same-ID prototype rows
prototype_id_weight is applied exactly once
```

EMA:

```text
memory changes after a warm forward pass
memory does not require gradients
projector parameters receive gradients
memory update happens after loss computation
```

Diagnostics:

```text
proto margins are finite after initialization
dead_slot_rate is in [0, 1]
effective_slots_per_id is in [1, prototype_per_id]
assignment_flip_rate is skipped when sample indices are unavailable
```

Checkpoint:

```text
resume restores projector parameters
resume restores or deliberately reinitializes memory
uninitialized zero banks are never treated as ready
```

## 23. Common Mistakes

- Copying host-specific architecture logic instead of writing a feature adapter.
- Initializing from one feature source and training on another.
- Initializing memory from raw host features but training loss on projected
  features.
- Building prototypes from the identity sampler's epoch stream instead of the
  full train dataset.
- Leaving text or image augmentation enabled during memory initialization.
- Using non-contiguous identity labels.
- Applying global nearest-prototype assignment during EMA updates.
- Updating memory before computing the loss.
- Letting EMA updates track gradients.
- Registering prototype banks as trainable parameters.
- Including same-ID rows in hard negatives.
- Forgetting the zero loss path before memory is initialized.
- Applying `prototype_id_weight` twice.
- Forgetting prototype projector parameters in the optimizer.
- Assuming prototype validation fusion is required for the training proof.
- Logging prototype diagnostics before memory is initialized.

## 24. Plug-and-Play Definition

The prototype branch is plug-and-play when:

1. The host exposes paired image/text embeddings and contiguous train labels.
2. A small adapter chooses prototype feature sources without changing the host
   architecture.
3. The branch owns its projection space and memory buffers.
4. Memory initialization uses a full train-set feature scan.
5. Training updates are identity-restricted and no-grad.
6. The host receives one extra weighted scalar loss.
7. Existing host evaluation remains valid without prototype-specific inference
   changes.

Under this contract, replacing the backbone changes the feature provider, not
the prototype algorithm.
