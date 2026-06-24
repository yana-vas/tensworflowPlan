# REBUILD_PLAN.md — 3DScan: Transformer Migration

**Project:** 3DScan — single 2D image → 3D STL mesh (3D-printable)
**Spec source:** `projectGoal.pdf` (Hristo Kalinov, Yana Vasileva)
**Authoring context:** Phase 1 reconnaissance of `code/3dScanSap/` + `Deep-Learning-With-Pytorch/` lecture notebooks.
**Status:** PLAN ONLY — no code is written in this phase.

---

## 0. Executive Summary

The existing project `code/3dScanSap/` is a **classic Occupancy Network (ONet, Mescheder et al. 2019)**: a CNN (ResNet‑18) compresses the image into a **single 256‑d global latent vector**, and an **MLP** predicts occupancy for each 3D point by *concatenating* that global vector with the point's `(x, y, z)`.

The spec (`projectGoal.pdf`) demands a fundamentally different, **transformer-native** design:

1. A **Vision Transformer encoder (DINOv2)** that splits the image into patches and emits a **sequence of patch tokens** (not a single vector).
2. A **cross-attention decoder** where each 3D coordinate is a **query** that attends directly to the patch tokens to decide occupancy.
3. The same back-end: 64³ grid query → threshold → Marching Cubes → Trimesh → STL.

So the **front two-thirds of the network are replaced**, while the data layer and mesh back-end are **largely reusable**. The key technical challenges the spec under-specifies — and that this plan resolves — are: (a) how continuous 3D coordinates become transformer queries (**Fourier/positional encoding**), (b) keeping the decoder **point-wise independent** so it scales to 262 K grid queries, (c) **evaluation rigor** (the spec defines no metrics), and (d) **watertight/printable** mesh output.

---

## 1. Current State — what `code/3dScanSap/` does, end to end

### 1.1 Pipeline (as built)

```
image (any size)
  └─ ImagePreprocessor: resize 224×224, ImageNet normalize          [src/data/preprocessing.py]
       └─ ResNetEncoder: ResNet-18 (ImageNet), fc→Linear(512,256)   [src/model/encoder.py]
            └─ latent vector  (B, 256)                              ← GLOBAL, spatial info lost
                 └─ OccupancyDecoder: 5×Linear+ReLU MLP             [src/model/decoder.py]
                      input = concat([latent broadcast, xyz]) (B,N,259)
                      └─ occupancy logits (B, N, 1)
                           └─ generate_occupancy_grid: 64³ grid     [src/model/occupancy_network.py]
                                └─ sigmoid → (64,64,64) prob grid
                                     └─ Marching Cubes (skimage)    [src/mesh/marching_cubes.py]
                                          └─ Trimesh → STL          [src/mesh/export.py]
```

### 1.2 Component facts (verified by reading every file)

| File | What it is | Key detail |
|---|---|---|
| `src/model/encoder.py` | `ResNetEncoder(latent_dim=256)` | **ResNet‑18** ImageNet; `fc` → `Linear(512,256)`; `freeze_backbone()`/`unfreeze_backbone()` helpers. Output **(B,256)**. **Not a transformer.** |
| `src/model/decoder.py` | `OccupancyDecoder(latent=256,hidden=256,layers=5)` | **MLP**: `Linear(259,256)→ReLU→3×(Linear+ReLU)→Linear(256,1)`. Conditioning = **concatenation** of broadcast latent with xyz. **No attention.** |
| `src/model/occupancy_network.py` | `OccupancyNetwork` | Wraps encoder+decoder. `generate_occupancy_grid(res=64, batch=100000)` builds `linspace(-1,1)³` grid, sigmoids, returns numpy `(R,R,R)`. `save/load/from_checkpoint`. |
| `src/data/dataset.py` | `ShapeNetDataset`, `get_dataloader` | 13 hardcoded categories. Reads `{cat}/{model}/points.npz` (`points`, bit-packed `occupancies` via `np.unpackbits`) and a **random view** from `{model}/img_choy2016/*.jpg`. 80/10/10 split, per-category `RandomState(42)`. Samples `num_points=2048`. This is the **Choy 2016 / OccNet data format.** |
| `src/data/point_sampling.py` | `PointSampler` | Surface+uniform sampling via `trimesh.sample_surface` + `mesh.contains`. **Data-prep utility only** — not used by the main `points.npz` path. |
| `src/data/preprocessing.py` | `ImagePreprocessor` | 224×224; train aug = resize+RandomCrop+HFlip+ColorJitter; ImageNet mean/std. |
| `src/mesh/marching_cubes.py` | `MarchingCubesExtractor` | `skimage.measure.marching_cubes(level=0.5)`, rescales verts `[0,res-1]→[-1,1]`, `fix_normals()`. |
| `src/mesh/export.py` | `MeshExporter`, `save_mesh` | Trimesh export STL (binary/ascii), OBJ, PLY. |
| `train.py` | training driver | `BCEWithLogitsLoss`, `Adam(lr=1e-4)`, `CosineAnnealingLR`, last/best/interrupted checkpoints, `--resume`, `--max-samples`. **No metrics beyond loss; no AMP, no grad clip, no warmup.** |
| `inference.py` | inference driver | checkpoint → image → grid → marching cubes → STL. |
| `src/utils/config.py` | `Config`/`ConfigDict` | YAML + attribute access; defaults mirror `configs/default.yaml`. |
| `configs/default.yaml` | hyperparams | `latent=256,hidden=256,layers=5,encoder=resnet18`; `bs=16,lr=1e-4,epochs=100,num_points=2048`; `grid=64,thr=0.5`; `image=224`. |

### 1.3 What works and is worth keeping

- **Data format & loader** (OccNet/Choy layout) — correct and reusable with minor extension.
- **Mesh back-end** (Marching Cubes + Trimesh STL export) — fully reusable.
- **Training scaffolding** (checkpointing, resume, config, CLI) — reusable, needs hardening.
- **Grid-batched inference** in `generate_occupancy_grid` — reusable pattern.

### 1.4 What violates the spec

- **Encoder is a CNN (ResNet‑18), not a ViT.** No patches, no tokens.
- **Decoder is concat-MLP, not cross-attention.** Points never "attend" to image regions.
- **A single global latent destroys spatial detail** — the exact failure mode patch-token cross-attention is meant to fix.

---

## 2. Target State — `projectGoal.pdf` mapped requirement-by-requirement

| # | Requirement (from spec) | Current status | Target |
|---|---|---|---|
| **R1** | Input: one 2D photo → Output: 3D mesh as **STL, 3D-printable** | Partial (STL yes; printability not guaranteed) | Keep STL; **add watertight/printable post-processing** |
| **R2** | **Step 1 — Vision Transformer encoder** (e.g. ready DINOv2); splits image into **patches → token vectors** | ✗ ResNet‑18 global vector | **DINOv2 ViT** → patch-token sequence `(B, N_patch, D)` |
| **R3** | **Step 2 — Cross-attention decoder**; 3D point = **query**; returns occupancy probability **0–1**; points **interact directly** with image vectors | ✗ concat-MLP | **Cross-attention decoder**: Fourier-encoded coords as queries; K/V = patch tokens; sigmoid head |
| **R4** | **Step 3 —** build **64³ grid**, threshold (e.g. 0.5), **Marching Cubes** → triangle mesh, **Trimesh** → STL | ✓ (reusable) | Keep; make resolution/threshold config-driven; optional padding/MISE |
| **R5** | Dataset: **ShapeNet, 13 categories**; images from many angles + occupancy npz | ✓ (loader exists) | Keep; **extend** to also load `pointcloud.npz` for evaluation |
| **R6** | *(Optional)* extra data via Blender (render + point sampling) | Partial (`point_sampling.py`) | Keep as optional offline tool; out of core scope |

**Spec omissions this plan fills (see §3):** evaluation metrics, coordinate query encoding, training-stability recipe, printable-mesh guarantee.

---

## 3. Recommendations (additions / modifications / removals — each justified)

### 3.1 ADD — Coordinate query encoding (Fourier features) — **essential, not optional**
The spec says "project the coordinates as queries" but not *how*. A raw 3-vector cannot be a meaningful `d_model`-dimensional attention query. Use **NeRF-style Fourier positional encoding**: `γ(p) = [p, sin(2⁰πp), cos(2⁰πp), …, sin(2^{L−1}πp), cos(2^{L−1}πp)]` then a `Linear → d_model`. With `L=10` octaves: `3 + 3·2·10 = 63 → d_model`.
**Why:** transformers/MLPs are spectrally biased toward low frequencies; Fourier features are the standard fix for learning high-frequency 3D geometry (NeRF, ONet variants). Without it the surface will be blobby. The lecture notebooks' `PositionalEncoding` (08_GPT) is the sinusoidal analog — we adapt the same idea to continuous coords.

### 3.2 ADD — A point-wise (Perceiver-style) decoder, **no self-attention among query points** — **architectural best practice**
Occupancy is a **point-wise field**; each query's answer is independent. Make the decoder **cross-attention only** (query→tokens) + FFN, with **no self-attention between the 2048 (train) / 262 144 (infer) points**.
**Why:** self-attention among points is O(N²) and would make 64³ inference intractable and pointless (points don't need each other). This matches the spec's wording ("3D points directly interact with the image vectors") and mirrors PerceiverIO / 3D-RETR decoding.

### 3.3 ADD — Evaluation module (the spec defines **zero** metrics) — **evaluation rigor**
Implement the standard single-view reconstruction metrics: **Volumetric IoU** (on occupancy grid), **Chamfer-L1 distance**, **Normal Consistency**, and **F-Score@τ**. Requires loading `pointcloud.npz` (surface points + normals) which the current loader ignores.
**Why:** BCE loss alone cannot tell you if shapes are *correct*. These are the ONet/3D-RETR benchmark metrics — needed to defend the project and compare against the ResNet baseline.

### 3.4 ADD — Transformer training-stability recipe — **training stability**
Adopt: **AdamW** (decoupled weight decay) + **linear warmup → cosine decay**, **gradient clipping** (`clip_grad_norm_(…, 1.0)`, exactly the notebooks' pattern in 6.3/7.1), **pre-norm** transformer blocks (LayerNorm before attention/FFN, as in 08_GPT), **dropout 0.1**, and **mixed precision (AMP)**.
**Why:** raw `Adam` + plain cosine (current `train.py`) is fragile for attention; warmup + clipping + pre-norm are the well-established stabilizers and are demonstrated in the course notebooks.

### 3.5 ADD — Freeze DINOv2 by default; optional LoRA fine-tune — **leverages notebook 05 + 11**
Stage 1: **freeze the ViT backbone**, train only the decoder + projections (transfer-learning pattern, notebook 5.1). Stage 2 (optional): **LoRA** adapters on the ViT attention `q/k/v` (notebook 11's `LoraConfig`) for cheap fine-tuning.
**Why:** DINOv2 features are already excellent; freezing avoids catastrophic forgetting on a small dataset and trains far faster. LoRA gives a controlled quality bump without full backbone updates.

### 3.6 ADD — Watertight / printable mesh post-processing — **satisfies R1 "3D-printable"**
After Marching Cubes: **pad the occupancy grid with a zero border** (so surfaces close at the bounding box), keep the **largest connected component**, run `trimesh` `fill_holes()` + `fix_normals()`, and verify `is_watertight`.
**Why:** the spec's explicit goal is a *printable* STL; the current code can emit open, non-manifold meshes. Border padding is a one-line fix that prevents the most common "open bottom" artifact.

### 3.7 ADD — Reusable transformer layer library + tests + logging
- `src/model/layers.py` holding `MultiHeadCrossAttention`, `FourierPositionalEncoding`, `FeedForward`, `CrossAttentionBlock` (adapted from notebooks 08_GPT / 09_BERT).
- **TensorBoard** logging of loss + metrics.
- **pytest** unit tests for shape contracts (already in `requirements.txt`).
**Why:** the notebooks already contain verified implementations of every block we need — centralizing them keeps the model files thin and testable.

### 3.8 MODIFY — Make `d_model` track the DINOv2 variant; config-drive everything
DINOv2 embed dims: **ViT‑S/14 = 384**, **ViT‑B/14 = 768**, ViT‑L/14 = 1024. With 224×224 input → patch 14 → **16×16 = 256 patch tokens**. Recommend **ViT‑S/14 (`d_model=384`, 6 heads)** as the default (good quality/compute trade-off), with the config exposing the variant and a learned `Linear` projection if decoder `d_model` differs.
**Why:** `latent_dim=256` is meaningless under the new design; decoder width must match (or project from) the token width.

### 3.9 MODIFY — Keep threshold 0.5 but clarify logit vs. probability; keep 64³ default
Threshold on the **sigmoid probability** (as the current grid code does). Keep 64³ as default infer resolution; expose 128³ for final renders. Optionally add **MISE** (multiresolution iso-surface extraction) later for sharper meshes at lower cost — mark as stretch goal, not core.

### 3.10 REMOVE / DEPRIORITIZE
- **`point_sampling.py`**: not on the training path (the dataset uses precomputed `points.npz`). **Keep as an optional offline data-prep tool** for the Blender path (R6); do not wire it into training.
- **Global-latent code paths** in encoder/decoder/occupancy_network: **delete** — superseded by tokens + cross-attention.
- **`encoder: resnet18` config key**: replace with `encoder: dinov2_vits14`.

---

## 4. Delta Analysis — exactly what changes

### 4.1 Rewritten (same path, new internals)
| File | Change |
|---|---|
| `src/model/encoder.py` | `ResNetEncoder` → **`DINOv2Encoder`** (patch-token output). |
| `src/model/decoder.py` | `OccupancyDecoder` (MLP) → **`CrossAttentionOccupancyDecoder`**. |
| `src/model/occupancy_network.py` | Wire token sequence through cross-attention; keep `generate_occupancy_grid`/`save`/`load` API, update internals. |
| `train.py` | Add AdamW, warmup+cosine, grad clip, AMP, metric logging, optional backbone freeze/LoRA flags. |
| `configs/default.yaml` | New model block (`encoder: dinov2_vits14`, `d_model`, `n_heads`, `n_layers`, `dropout`, `fourier_bands`); training block (`warmup_steps`, `weight_decay`, `grad_clip`, `freeze_backbone`, `amp`). |
| `src/utils/config.py` | Update `DEFAULTS` to match new config schema. |
| `requirements.txt` | Add transformer/eval deps (§7). |
| `README.md` | Replace stub with architecture + usage. |

### 4.2 New files
| File | Why |
|---|---|
| `src/model/layers.py` | Reusable `MultiHeadCrossAttention`, `FourierPositionalEncoding`, `FeedForward`, `CrossAttentionBlock`. |
| `src/eval/metrics.py` | IoU, Chamfer-L1, Normal Consistency, F-Score. |
| `src/eval/evaluate.py` | Eval driver over the test split → metrics table. |
| `src/mesh/postprocess.py` | Watertight/largest-component/fill-holes for printability. |
| `tests/test_shapes.py` | Shape-contract unit tests for encoder/decoder/end-to-end. |
| `configs/dinov2_vitb.yaml` | Optional larger-model config (ViT‑B/14, d_model=768). |

### 4.3 Extended (not rewritten)
| File | Change |
|---|---|
| `src/data/dataset.py` | Also load `pointcloud.npz` (points+normals) when `return_eval=True`; optional camera load from `img_choy2016/cameras.npz`. Keep splits, categories, image loading. |
| `src/mesh/marching_cubes.py` | Add optional zero-border padding before `marching_cubes`; call `postprocess` hook. |
| `inference.py` | Call mesh post-processing; report watertightness. |

### 4.4 Deleted
| File | Reason |
|---|---|
| ResNet code inside `encoder.py` | Replaced by DINOv2. |
| MLP concat code inside `decoder.py` | Replaced by cross-attention. |
| `point_sampling.py` | **Not deleted** — relocated conceptually to optional `tools/`; excluded from the model package. (Physical move optional.) |

---

## 5. Transformer Migration — per non-transformer component

### 5.1 Encoder: ResNet‑18 → **DINOv2 Vision Transformer**

| Aspect | Old | New |
|---|---|---|
| Model class | `ResNetEncoder` (torchvision `resnet18`) | `DINOv2Encoder` (`torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` **or** HF `AutoModel.from_pretrained('facebook/dinov2-small')`) |
| Output | global vector `(B,256)` | **patch tokens `(B, 256, 384)`** (16×16 patches, ViT‑S/14) |
| Attention | none (conv) | ViT **multi-head self-attention** (pretrained, frozen by default) |
| Positional enc. | n/a | **DINOv2 learned, interpolatable** patch position embeddings (built in) |
| Patchify / "tokenization" | n/a | 14×14 conv patch embedding (built in); 224/14 = 16 → 256 tokens |
| Projection | `Linear(512,256)` | optional `Linear(384, d_model)` only if decoder `d_model ≠ 384` |
| Normalization | ImageNet mean/std | **same** ImageNet mean/std (DINOv2 expects it) — current `ImagePreprocessor` is compatible; image side must be a multiple of 14 (224 ✓) |

Use `forward_features(x)['x_norm_patchtokens']` (hub) to get patch tokens; drop/keep the CLS token (recommend **keep CLS** appended to the token set — gives the decoder a global summary alongside local patches).

### 5.2 Decoder: concat-MLP → **Cross-Attention Occupancy Decoder**

| Aspect | Old | New |
|---|---|---|
| Model class | `OccupancyDecoder` (MLP) | `CrossAttentionOccupancyDecoder` |
| Query | raw `xyz` concatenated to latent | **Fourier-encoded `xyz` → Linear → `d_model`** (one token per point) |
| Keys/Values | n/a | **patch tokens** from DINOv2 `(B, N_patch, d_model)` |
| Attention mechanism | none | **multi-head cross-attention** (query=points, K/V=tokens), scaled dot-product, `n_heads=6`, adapted from notebook 09_BERT `MultiHeadAttention` (non-causal) |
| Self-attention among points | n/a | **none** (point-wise independence → scalable to 262 K queries) |
| Block structure | `Linear+ReLU` ×5 | `n_layers` × **pre-norm `CrossAttentionBlock`**: `x = x + CrossAttn(LN(x), tokens)`; `x = x + FFN(LN(x))` (GELU, ×4 expand) — from notebook 08_GPT `TransformerBlock` |
| Positional encoding | implicit raw coords | **Fourier features** (NeRF-style); tokens already carry DINOv2 pos-embed |
| Output head | `Linear(256,1)` | `LayerNorm → Linear(d_model,1)` → logit; sigmoid at inference |
| Loss compatibility | `BCEWithLogitsLoss` | **unchanged** |

**Reference implementation source:** the cross-attention class is the notebook `MultiHeadAttention` (09_BERT) with **separate Q vs. K/V inputs** — the exact `CrossAttention` template the notebook recon already sketched. The `PositionalEncoding`/`TransformerBlock`/pre-norm residual pattern is from notebook 08_GPT.

### 5.3 OccupancyNetwork wrapper
- `forward(images, points)` API **unchanged** (drop-in for `train.py`).
- Internals: `tokens = encoder(images)` → `logits = decoder(points, tokens)`.
- `generate_occupancy_grid(...)`: **encode image once**, reuse tokens across all grid batches (current code already caches the latent once — same pattern, tokens instead of vector).

### 5.4 Tokenization note
There is **no text tokenization** in this project. "Tokens" = image patch embeddings (ViT) and per-point coordinate embeddings. No vocabulary, no BPE.

---

## 6. File-by-File Blueprint (rebuilt project)

> Package root: `code/3dScanSap/`. Requirement IDs refer to §2.

### `src/model/layers.py` *(new)* — satisfies R2, R3
- `FourierPositionalEncoding(num_bands=10, include_input=True)` — `forward(coords (B,N,3)) → (B,N,3+6·num_bands)`.
- `MultiHeadCrossAttention(d_model, n_heads, dropout)` — `forward(query (B,Nq,d), kv (B,Nk,d), key_padding_mask=None) → (B,Nq,d)`. Scaled dot-product, head split/merge.
- `FeedForward(d_model, mult=4, dropout)` — GELU MLP.
- `CrossAttentionBlock(d_model, n_heads, dropout)` — pre-norm cross-attn + FFN with residuals.

### `src/model/encoder.py` *(rewrite)* — satisfies R2
- `DINOv2Encoder(variant='dinov2_vits14', out_dim=None, freeze=True)`.
- `forward(images (B,3,224,224)) → tokens (B, N_patch(+1), embed_dim or out_dim)`.
- `freeze_backbone()/unfreeze_backbone()`; optional `apply_lora(r, alpha, targets)`.
- Exposes `embed_dim`, `num_tokens`.

### `src/model/decoder.py` *(rewrite)* — satisfies R3
- `CrossAttentionOccupancyDecoder(d_model, n_heads=6, n_layers=4, num_bands=10, dropout=0.1, token_dim=None)`.
- Submodules: `FourierPositionalEncoding`, `Linear(fourier_dim → d_model)`, optional `Linear(token_dim → d_model)`, `ModuleList[CrossAttentionBlock]`, `LayerNorm`, `Linear(d_model,1)`.
- `forward(points (B,N,3), tokens (B,Nk,token_dim)) → logits (B,N,1)`.

### `src/model/occupancy_network.py` *(rewrite)* — satisfies R2,R3,R4
- `OccupancyNetwork(encoder_cfg, decoder_cfg)`.
- `forward(images, points) → logits (B,N,1)`.
- `generate_occupancy_grid(images, resolution=64, batch_size=100000) → np.ndarray (R,R,R)` (sigmoid, single image encode).
- `save/load/from_checkpoint` (store full model + encoder/decoder cfg).

### `src/data/dataset.py` *(extend)* — satisfies R5, R3-eval
- Keep `ShapeNetDataset`, `CATEGORIES`, `get_dataloader`, splits.
- New flag `return_eval`: also load `pointcloud.npz` → `{'eval_points','eval_normals'}` for metrics.
- Optional `load_camera` → camera extrinsics/intrinsics from `cameras.npz`.

### `src/data/preprocessing.py` *(keep)* — R2
- `ImagePreprocessor` unchanged (ImageNet norm is DINOv2-correct). Add assert that `image_size % 14 == 0`.

### `src/eval/metrics.py` *(new)* — satisfies §3.3 (eval rigor)
- `volumetric_iou(pred_occ, gt_occ, threshold=0.5) → float`.
- `chamfer_l1(pred_points, gt_points) → float`.
- `normal_consistency(pred_mesh, gt_points, gt_normals) → float`.
- `f_score(pred_points, gt_points, tau) → float`.

### `src/eval/evaluate.py` *(new)*
- Loads checkpoint, iterates test split, produces per-category + mean metric table; CLI `--checkpoint --data-root --resolution`.

### `src/mesh/marching_cubes.py` *(extend)* — R4
- Add `pad=True` (zero border) before `measure.marching_cubes`; keep rescale/`fix_normals`; call postprocess.

### `src/mesh/postprocess.py` *(new)* — satisfies §3.6 (R1 printable)
- `make_printable(mesh) → mesh`: largest component, `fill_holes`, `fix_normals`, watertight check + report.

### `src/mesh/export.py` *(keep)* — R4
- `MeshExporter`/`save_mesh` unchanged.

### `train.py` *(rewrite)* — R2,R3,R5 + §3.4/§3.5
- AdamW, **linear warmup → cosine**, `clip_grad_norm_`, **AMP (`autocast`+`GradScaler`)**, `--freeze-backbone/--lora`, TensorBoard, periodic IoU on val, best/last/interrupted checkpoints, `--resume`.

### `inference.py` *(extend)* — R1,R4
- image → tokens → grid → marching cubes → **`make_printable`** → STL; report watertightness.

### `src/utils/config.py` + `configs/default.yaml` *(update)*
- New schema (encoder variant, d_model, n_heads, n_layers, dropout, fourier_bands; warmup, weight_decay, grad_clip, freeze_backbone, amp; grid, threshold).

### `tests/test_shapes.py` *(new)*
- Asserts encoder token shape, decoder logit shape, end-to-end `(B,N,1)`, grid `(R,R,R)`, Fourier dim.

---

## 7. Dependency Manifest

```text
# --- core (existing, keep; align to course: torch 2.x) ---
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24,<2.0          # <2.0 for trimesh/skimage compatibility
scipy>=1.10.0

# --- transformer encoder (DINOv2) ---
# Option A (recommended): torch.hub — no extra dep beyond torch; xformers optional for speed
xformers>=0.0.23          # optional, accelerates DINOv2 attention (skip on CPU-only)
# Option B: HuggingFace
transformers>=4.40.0      # AutoModel 'facebook/dinov2-small'
timm>=0.9.16              # optional ViT utilities
einops>=0.7.0             # clean tensor reshapes in attention

# --- optional efficient fine-tuning (notebook 11) ---
peft>=0.10.0              # LoRA on DINOv2 (optional)

# --- mesh / 3D (existing) ---
scikit-image>=0.21.0      # marching cubes
trimesh>=4.0.0
rtree>=1.0.0              # trimesh.contains / proximity
pykdtree>=1.3.0           # fast NN for Chamfer/F-score (or scipy.cKDTree)

# --- data / image (existing) ---
Pillow>=9.5.0
PyYAML>=6.0

# --- training / logging / utils ---
tqdm>=4.65.0
tensorboard>=2.15.0
matplotlib>=3.7.0

# --- tests ---
pytest>=7.3.0
```

Notes: prefer **torch.hub DINOv2** (fewer deps, matches course's `torchvision`/hub style); fall back to `transformers` if hub access is blocked. Pin `numpy<2.0` to avoid trimesh/skimage breakage. CUDA build per course (`cu12x`).

---

## 8. Migration Sequence (ordered; dependencies explicit)

1. **Lock dependencies** — update `requirements.txt` (§7); verify DINOv2 loads via `torch.hub` and emits `(B,256,384)` tokens for a 224² input. *(Blocks everything ViT.)*
2. **Build `src/model/layers.py`** — Fourier PE, cross-attention, FFN, block; unit-test shapes. *(Depends: 1. Blocks decoder.)*
3. **Rewrite `encoder.py` → `DINOv2Encoder`** — token output + freeze/LoRA; shape test. *(Depends: 1.)*
4. **Rewrite `decoder.py` → `CrossAttentionOccupancyDecoder`** — consumes tokens + Fourier queries. *(Depends: 2,3.)*
5. **Rewrite `occupancy_network.py`** — wire encoder→decoder; keep `generate_occupancy_grid`/`save`/`load` API. *(Depends: 3,4.)*
6. **Update `config.py` + `default.yaml`** to new schema. *(Depends: 3–5.)*
7. **Extend `dataset.py`** — add `pointcloud.npz`/camera loading behind flags; keep splits/loader. *(Independent of 2–6; can parallelize with steps 2–5.)*
8. **Harden `train.py`** — AdamW, warmup+cosine, grad clip, AMP, freeze/LoRA flags, TensorBoard, val IoU. Smoke-train with `--max-samples` on one category. *(Depends: 5,6,7.)*
9. **Add `src/eval/metrics.py` + `evaluate.py`** — IoU/Chamfer/NC/F-score over test split. *(Depends: 5,7.)*
10. **Add `src/mesh/postprocess.py`; extend `marching_cubes.py`** (padding) **and `inference.py`** (printable STL). *(Depends: 5.)*
11. **`tests/test_shapes.py`** green end-to-end (random image → STL). *(Depends: 5,10.)*
12. **Baseline comparison** — train transformer model + (optionally) retain old ResNet baseline; report metric deltas. Tune `n_layers`, `num_bands`, ViT‑S vs ViT‑B. *(Depends: 8,9.)*
13. **Docs** — rewrite `README.md` (architecture diagram, train/infer/eval commands). *(Depends: all.)*

**Critical path:** 1 → 2/3 → 4 → 5 → 8 → 12. Steps 7, 9, 10, 11 branch off step 5 and can be developed in parallel.

---

## 9. Open Questions for Phase-2 Review

1. **DINOv2 source:** `torch.hub` (recommended) vs HuggingFace `transformers`? Affects deps and offline availability.
2. **Backbone size:** ViT‑S/14 (`d_model=384`, fast) default vs ViT‑B/14 (`768`, sharper) — compute budget?
3. **Keep ResNet baseline** for an ablation table, or hard-delete? (Recommend keep behind a config flag for the comparison in step 12.)
4. **Camera conditioning:** canonical-frame occupancy (ONet-style, ignores view — simpler) vs feed camera pose (uses `cameras.npz`)? Recommend canonical first.
5. **MISE multiresolution extraction** — include now or as a later stretch goal? (Recommend later.)

---

*End of plan. No code written. Awaiting review/approval before Phase 2.*
