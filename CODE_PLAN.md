# CODE_PLAN.md — 3DScan (DINOv2 + Cross-Attention Occupancy)

> **Full engineering build guide.** Two developers can read this top-to-bottom, split the work, and build the entire transformer-based 3D reconstruction system. Every file ships with: a concept lesson (tied to the lecture notebooks), complete production code, a line-by-line walkthrough, and verification commands.

This document implements the architecture approved in [`REBUILD_PLAN.md`](./REBUILD_PLAN.md): a single 2D image → **DINOv2 ViT encoder** (patch tokens) → **cross-attention occupancy decoder** (3D coords as queries) → 64³ grid → Marching Cubes → printable STL.

---

## The Interface Contract (read this first — it is law)

Every file below conforms to these shapes and names. If you change one, change it everywhere.

| Symbol | Value / Shape | Defined in | Consumed by |
|---|---|---|---|
| Image batch | `(B, 3, 224, 224)` float32, ImageNet-normalized | `preprocessing.py` | `encoder.py` |
| `image_size` | `224` (must satisfy `image_size % 14 == 0`) | `config.data.image_size` | preprocessing, encoder |
| Encoder backbone | `dinov2_vits14` (embed_dim **384**, patch 14) | `config.model.encoder.variant` | `encoder.py` |
| **Patch tokens** | `(B, 257, 384)` = 1 CLS + 256 patches | `encoder.py` → `DINOv2Encoder.forward` | `decoder.py`, `occupancy_network.py` |
| `token_dim` | `384` (= encoder `embed_dim`) | `encoder.py` `.embed_dim` | `decoder.py` |
| `d_model` | `384` | `config.model.decoder.d_model` | `layers.py`, `decoder.py` |
| `n_heads` | `6` (384 / 6 = 64 per head) | `config.model.decoder.n_heads` | `layers.py`, `decoder.py` |
| `n_layers` | `4` cross-attention blocks | `config.model.decoder.n_layers` | `decoder.py` |
| `num_bands` | `10` Fourier bands → coord dim `3 + 3·2·10 = 63` | `config.model.decoder.num_bands` | `layers.py`, `decoder.py` |
| Query points | `(B, N, 3)` in `[-1, 1]³` | `dataset.py` / grid | `decoder.py`, `occupancy_network.py` |
| Decoder output | `(B, N, 1)` **logits** (no sigmoid) | `decoder.py` → `occupancy_network.forward` | `train.py` (BCEWithLogits) |
| Occupancy grid | `(R, R, R)` **probabilities** ∈ [0,1] | `occupancy_network.generate_occupancy_grid` | `marching_cubes.py` |
| `grid_resolution` | `64` | `config.inference.grid_resolution` | inference, eval |
| `threshold` | `0.5` | `config.inference.threshold` | marching cubes |

**ViT-B override** (`configs/dinov2_vitb.yaml`): `variant: dinov2_vitb14`, `token_dim/d_model: 768`, `n_heads: 12`, tokens `(B, 257, 768)`. Because the decoder always projects `token_dim → d_model` and `n_heads` divides `d_model` (768/12 = 64), nothing else changes.

---

## Task Assignment Table

| # | File | Dev | Complexity | Depends on |
|---|---|---|---|---|
| 1 | `src/__init__.py` | B | low | — |
| 2 | `src/model/__init__.py` | A | low | layers, encoder, decoder, occupancy_network |
| 3 | `src/model/layers.py` | A | **high** | — |
| 4 | `src/model/encoder.py` | A | medium | `torch.hub` (DINOv2) |
| 5 | `src/model/decoder.py` | A | **high** | layers.py |
| 6 | `src/model/occupancy_network.py` | A | **high** | encoder.py, decoder.py |
| 7 | `src/data/__init__.py` | B | low | dataset |
| 8 | `src/data/preprocessing.py` | B | low | — |
| 9 | `src/data/dataset.py` | B | medium | preprocessing.py |
| 10 | `src/mesh/__init__.py` | B | low | marching_cubes, postprocess, export |
| 11 | `src/mesh/export.py` | B | low | — |
| 12 | `src/mesh/postprocess.py` | B | medium | — |
| 13 | `src/mesh/marching_cubes.py` | B | medium | postprocess.py |
| 14 | `src/eval/__init__.py` | B | low | metrics |
| 15 | `src/eval/metrics.py` | B | **high** | — |
| 16 | `src/eval/evaluate.py` | B | medium | occupancy_network, dataset, metrics, mesh, config |
| 17 | `src/utils/__init__.py` | B | low | config |
| 18 | `src/utils/config.py` | B | medium | — |
| 19 | `configs/default.yaml` | B | low | — |
| 20 | `configs/dinov2_vitb.yaml` | B | low | — |
| 21 | `train.py` | A | **high** | occupancy_network, config, dataset |
| 22 | `inference.py` | A | medium | occupancy_network, mesh, config, preprocessing |
| 23 | `tests/__init__.py` | A | low | — |
| 24 | `tests/test_shapes.py` | A | **high** | layers, decoder, occupancy_network, mesh, metrics |
| 25 | `verify_setup.py` | A | low | `torch.hub` |
| 26 | `requirements.txt` | B | low | — |
| 27 | `README.md` | B | low | everything |

**Workload balance:** Dev A owns the transformer model + training/inference path (fewer files, higher complexity). Dev B owns data + mesh + eval + config + infra (more files, lower complexity each, but `metrics.py` is hard). Effort is roughly equal.

---

## Parallel Timeline

```
Day 1:  Dev A: layers.py, encoder.py            |  Dev B: preprocessing.py, dataset.py
Day 2:  Dev A: decoder.py, occupancy_network.py |  Dev B: config.py, default.yaml, dinov2_vitb.yaml
        ── SYNC 1: merge model+config; run verify_setup.py + layer/decoder shape checks ──
Day 3:  Dev A: train.py                          |  Dev B: export.py, postprocess.py, marching_cubes.py, metrics.py
        ── SYNC 2: smoke-train (--max-samples 8); confirm loss decreases ──
Day 4:  Dev A: inference.py, tests/test_shapes.py|  Dev B: evaluate.py, README.md, requirements.txt
        ── SYNC 3: full pipeline — train → evaluate → inference → open STL ──
```

### Integration Checkpoints

**SYNC 1 (end of Day 2) — Model + Config compile and shapes are correct.**
```bash
python verify_setup.py
python -c "import torch; from src.model import OccupancyNetwork; from src.utils.config import load_config; \
cfg=load_config('configs/default.yaml'); m=OccupancyNetwork.from_config(cfg); \
img=torch.randn(2,3,224,224); pts=torch.rand(2,128,3)*2-1; \
print('logits', m(img, pts).shape)"
# Expect: logits torch.Size([2, 128, 1])
```

**SYNC 2 (end of Day 3) — Training loop runs and learns.**
```bash
python train.py --config configs/default.yaml --data-root <DATA> --max-samples 8 --epochs 3
# Expect: 3 epochs print; train loss decreases monotonically-ish; checkpoints/last.pt written.
```

**SYNC 3 (end of Day 4) — End-to-end.**
```bash
pytest tests/ -v                       # all green
python inference.py --checkpoint checkpoints/best.pt --input <IMG> --output out.stl
python src/eval/evaluate.py --checkpoint checkpoints/best.pt --data-root <DATA> --max-samples 16
# Expect: out.stl opens in a mesh viewer and is watertight; eval prints an IoU/Chamfer/NC/F-score table.
```

---

# Overview

The system is three stages chained end-to-end:

```
                ┌────────────────────────────────────────────────────────────┐
   image  ──►   │  DINOv2Encoder (frozen ViT-S/14)                            │
 (B,3,224,224)  │    patch embedding → 12 self-attention blocks              │
                │    → tokens (B, 257, 384)   [1 CLS + 16×16 patches]         │
                └───────────────────────────────┬────────────────────────────┘
                                                 │ K, V
   points ──► FourierPE(63) ──► Linear→384 ──►  Q
 (B,N,3)        ┌───────────────────────────────▼────────────────────────────┐
                │  CrossAttentionOccupancyDecoder                            │
                │    4× [ pre-norm cross-attn(Q=points, K/V=tokens) + FFN ]  │
                │    → LayerNorm → Linear(384,1)                             │
                │    → logits (B, N, 1)                                      │
                └───────────────────────────────┬────────────────────────────┘
                          train: BCEWithLogitsLoss│   infer: sigmoid
                                                 ▼
                64³ grid query → prob grid (64,64,64) → Marching Cubes
                → make_printable (watertight) → Trimesh → STL
```

**What we borrow from the course, file by file:** the attention math is the `MultiHeadAttention` from `09_BERT/9.2_BERT_Text_Classification.ipynb`; the pre-norm residual block and sinusoidal `PositionalEncoding` are from `08_GPT/8.1_GPT.ipynb`; the frozen-backbone strategy is from `05_Transfer_learining_Fine-tuining/5.1 Transfer learning.ipynb`; the gradient-clipping training loop is from `06_Word2Vec_RNN_Text_Classification/6.3_Text Classification.ipynb` and `07_Language_Modeling_Token_Classification/7.1_Language_Modeling.ipynb`; optional LoRA is from `11_LLM_Fine-tuning/LLM_Fine_tuning.ipynb`; the DataLoader/transform split is from `04_CNN_Data_Augmentation/4.2 Data Augmentation.ipynb`.

---

# Model Core

## `src/__init__.py`

### Section 1: Header
- **Path:** `src/__init__.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Marks `src/` as a Python package so `from src.model import ...` resolves. Intentionally empty (no import side-effects, so partially-built subpackages don't break collaborators' imports during parallel development).

### Section 2: Concept
A directory becomes an importable package when it contains `__init__.py`. Keeping the top-level one empty avoids the classic parallel-dev hazard: if `src/__init__.py` eagerly imported `src.model`, then Dev B couldn't import `src.data` until Dev A's model code existed. Empty top-level `__init__` = maximum independence.

### Section 3: Code
```python
"""3DScan — single-image 3D reconstruction with DINOv2 + cross-attention occupancy."""
```

### Section 4: Walkthrough
- One module docstring, no imports. This guarantees `import src` never fails regardless of which subpackages are finished.

### Section 5: Verification
```bash
python -c "import src; print('ok')"
# Expect: ok
```

---

## `src/model/__init__.py`

### Section 1: Header
- **Path:** `src/model/__init__.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** Public API of the model package. Re-exports the three top-level classes so the rest of the codebase imports them as `from src.model import OccupancyNetwork, DINOv2Encoder, CrossAttentionOccupancyDecoder`.

### Section 2: Concept
A package `__init__.py` is the right place to define a *curated* public surface. `train.py`/`inference.py` should not need to know that `OccupancyNetwork` lives in `occupancy_network.py` — they import from the package. This is the same idea as `from torchvision.models import resnet18` hiding the file layout (seen throughout `05 Transfer learning`).

### Section 3: Code
```python
"""Model package: DINOv2 encoder + cross-attention occupancy decoder."""

from src.model.layers import (
    FourierPositionalEncoding,
    MultiHeadCrossAttention,
    FeedForward,
    CrossAttentionBlock,
)
from src.model.encoder import DINOv2Encoder
from src.model.decoder import CrossAttentionOccupancyDecoder
from src.model.occupancy_network import OccupancyNetwork

__all__ = [
    "FourierPositionalEncoding",
    "MultiHeadCrossAttention",
    "FeedForward",
    "CrossAttentionBlock",
    "DINOv2Encoder",
    "CrossAttentionOccupancyDecoder",
    "OccupancyNetwork",
]
```

### Section 4: Walkthrough
- **Absolute imports** (`from src.model.layers ...`) not relative — consistent with `train.py`'s `from src.model import OccupancyNetwork` and avoids ambiguity when files are run as scripts.
- `__all__` documents the supported API and controls `from src.model import *`.
- This file is written **last** among the model files (it imports all of them), but it is trivial — Dev A creates it after `occupancy_network.py` exists.

### Section 5: Verification
```bash
python -c "from src.model import OccupancyNetwork, DINOv2Encoder, CrossAttentionOccupancyDecoder; print('ok')"
# Expect: ok   (after files 3–6 exist)
```

---

## `src/model/layers.py`

### Section 1: Header
- **Path:** `src/model/layers.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The reusable transformer primitives. Four classes: `FourierPositionalEncoding` (turns continuous xyz into a high-frequency embedding), `MultiHeadCrossAttention` (queries attend to key/value tokens), `FeedForward` (the position-wise MLP), and `CrossAttentionBlock` (pre-norm residual wrapper combining the two). Everything in `decoder.py` is built from these.

### Section 2: Concept Lesson

**Fourier positional encoding — why raw xyz fails.**
In `08_GPT/8.1_GPT.ipynb` the `PositionalEncoding` class injected *discrete sequence position* using sinusoids of geometrically-spaced frequencies:
```python
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```
We borrow that *exact mathematical idea* — represent a coordinate as a bank of sines and cosines at multiple frequencies — but adapt it from discrete token indices to **continuous 3D coordinates**.

Why is this necessary? A linear layer (and, by extension, attention's linear projections) is biased toward learning **low-frequency** functions of its input — this is called **spectral bias** (Rahaman et al. 2019; Tancik et al. "Fourier Features", 2020). If we feed the decoder raw `(x, y, z)` ∈ [-1, 1], it can easily learn smooth, blobby occupancy fields but struggles to represent sharp surfaces, thin structures, and high-frequency detail. By first mapping each coordinate through `γ(p) = [p, sin(2⁰πp), cos(2⁰πp), …, sin(2^{L-1}πp), cos(2^{L-1}πp)]`, we hand the network a basis in which high-frequency geometry is *linearly accessible*. This is the same trick NeRF uses, and it is the single most important reason this decoder will produce crisp meshes instead of mush. With `L = num_bands = 10`, each of the 3 axes contributes `2·10` sinusoids plus its raw value → `3 + 3·2·10 = 63` dims.

**Multi-head cross-attention — borrowed from BERT, adapted for two inputs.**
`09_BERT/9.2_BERT_Text_Classification.ipynb` implements `MultiHeadAttention` where `q, k, v` are projected from the *same* sequence (self-attention). We adapt it into **cross-attention** by feeding **two** sequences: the query comes from the points, the key/value come from the image tokens. Mechanically (identical to the notebook): project to Q, K, V; split into heads; `softmax(QKᵀ/√d_k)`; weight V; merge heads; output projection. The only change is the call signature `forward(query, kv)` instead of `forward(x)`. We use PyTorch's fused `F.scaled_dot_product_attention` instead of the notebook's manual `softmax(q @ k.transpose / sqrt(d_k)) @ v` — same math, but it dispatches to FlashAttention kernels on GPU (the notebook wrote it by hand for teaching; production uses the fused op).

**FeedForward + pre-norm block — borrowed from GPT.**
The `TransformerBlock` in `08_GPT/8.1_GPT.ipynb` is `x = ln1(x + attn(x)); x = ln2(x + ffn(x))` with `ffn = Linear(d, 4d) → GELU → Linear(4d, d)`. That is *post-norm* in the notebook's exact form (LayerNorm applied to the sum). We use the now-standard **pre-norm** variant (`x = x + attn(ln(x))`) because it is markedly more stable to train for deep stacks — gradients flow through the residual path unimpeded. We keep the 4× expansion and GELU exactly as the notebook taught.

**What's borrowed / adapted / new:**
- *Borrowed:* sinusoidal frequency banks (08_GPT PE), scaled-dot-product multi-head attention (09_BERT), 4×-GELU FFN (08_GPT).
- *Adapted:* PE → continuous coords; self-attention → cross-attention (Q vs K/V split); post-norm → pre-norm.
- *New:* point queries are independent (no self-attention among them — see `decoder.py`), and `F.scaled_dot_product_attention` for fused kernels.

### Section 3: The Complete Code
```python
"""Reusable transformer primitives for the occupancy decoder.

Borrows the attention math from 09_BERT/9.2 and the FFN/positional-encoding
ideas from 08_GPT/8.1, adapted for continuous 3D coordinates and cross-attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    """Map continuous coordinates to a multi-frequency sin/cos embedding.

    Adapts the sinusoidal ``PositionalEncoding`` from 08_GPT/8.1 (which encoded
    discrete sequence positions) to continuous 3D coordinates, defeating the
    spectral bias of linear layers (Tancik et al. 2020).

    For input of last-dim ``in_dim`` and ``num_bands`` frequency bands the output
    last-dim is ``(in_dim if include_input else 0) + in_dim * 2 * num_bands``.
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_bands: int = 10,
        include_input: bool = True,
    ) -> None:
        """Build the (non-trainable) frequency bank.

        Args:
            in_dim: Number of coordinate axes (3 for xyz).
            num_bands: Number of octaves; frequencies are ``2^k * pi``.
            include_input: If True, concatenate the raw coordinates as well.
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.include_input = include_input

        # Frequencies 2^0..2^(L-1) scaled by pi, as in NeRF. Registered as a
        # buffer (not a Parameter): fixed constants, but must follow .to(device)
        # and be saved in the state_dict for reproducibility.
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=True)

        self.output_dim = (in_dim if include_input else 0) + in_dim * 2 * num_bands

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.

        Args:
            coords: ``(..., in_dim)`` coordinates, typically in ``[-1, 1]``.

        Returns:
            ``(..., output_dim)`` Fourier features.
        """
        # coords: (..., D) -> (..., D, 1) * freqs (F,) -> (..., D, F)
        scaled = coords.unsqueeze(-1) * self.freqs
        # (..., D, F) -> (..., D, 2F) -> (..., D*2F)
        sincos = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = sincos.flatten(start_dim=-2)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key/value tokens.

    Same projection/head-split/scaled-dot-product/merge structure as the
    ``MultiHeadAttention`` in 09_BERT/9.2, but Q is projected from a different
    input than K and V, making it *cross*-attention. Uses the fused
    ``F.scaled_dot_product_attention`` (same math as the notebook's manual
    softmax, but dispatches to FlashAttention kernels).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """Create the four linear projections.

        Args:
            d_model: Model width; must be divisible by ``n_heads``.
            n_heads: Number of attention heads.
            dropout: Attention dropout probability (applied inside SDPA).
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Attend ``query`` over ``kv``.

        Args:
            query: ``(B, Nq, d_model)`` query tokens (the 3D points).
            kv: ``(B, Nk, d_model)`` key/value tokens (the image patches).

        Returns:
            ``(B, Nq, d_model)`` attended features.
        """
        b, nq, _ = query.shape
        nk = kv.shape[1]

        # Project then split into heads: (B, N, d_model) -> (B, n_heads, N, d_k)
        q = self.q_proj(query).view(b, nq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(kv).view(b, nk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(kv).view(b, nk, self.n_heads, self.d_k).transpose(1, 2)

        # Fused scaled-dot-product attention (softmax(QK^T/sqrt(d_k)) V).
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )  # (B, n_heads, Nq, d_k)

        # Merge heads back: (B, n_heads, Nq, d_k) -> (B, Nq, d_model)
        attn = attn.transpose(1, 2).contiguous().view(b, nq, self.d_model)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Dropout -> Linear.

    Identical structure to the FFN inside 08_GPT/8.1's ``TransformerBlock``
    (``Linear(d, mult*d) -> GELU -> Linear(mult*d, d)``).
    """

    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1) -> None:
        """Args:
        d_model: Model width.
        mult: Hidden expansion factor (4 in the notebook and the original Transformer).
        dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the FFN. Args: x ``(..., d_model)``. Returns ``(..., d_model)``."""
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm residual block: cross-attention then feed-forward.

    Same residual+LayerNorm+FFN skeleton as 08_GPT/8.1's ``TransformerBlock``,
    but (a) the attention is cross-attention over external ``kv`` tokens and
    (b) normalization is *pre*-norm (``x + sublayer(norm(x))``) for training
    stability in deep stacks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        """Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout used in attention and FFN.
        ffn_mult: FFN hidden expansion factor.
        """
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Args:
        x: ``(B, Nq, d_model)`` query stream (points).
        kv: ``(B, Nk, d_model)`` image tokens (already projected to d_model).

        Returns:
            ``(B, Nq, d_model)``.
        """
        x = x + self.attn(self.norm_q(x), self.norm_kv(kv))
        x = x + self.ffn(self.norm_ff(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    # FourierPE
    pe = FourierPositionalEncoding(in_dim=3, num_bands=10)
    pts = torch.rand(2, 128, 3) * 2 - 1
    enc = pe(pts)
    assert enc.shape == (2, 128, 63), enc.shape
    assert pe.output_dim == 63

    # Cross-attention block end-to-end
    block = CrossAttentionBlock(d_model=384, n_heads=6, dropout=0.0)
    q = torch.randn(2, 128, 384)
    kv = torch.randn(2, 257, 384)
    out = block(q, kv)
    assert out.shape == (2, 128, 384), out.shape
    print("layers.py self-test passed:",
          "fourier", tuple(enc.shape), "block", tuple(out.shape))
```

### Section 4: Line-by-Line Walkthrough
- **`register_buffer("freqs", ...)` not `nn.Parameter`.** The frequencies are fixed constants of the encoding, not things we learn. A buffer is excluded from `optimizer.param_groups` (never updated by gradient descent) yet still moves with `model.to(device)` and is saved/loaded in the `state_dict`. Making it a `Parameter` would let training drift the frequencies and break reproducibility; making it a plain tensor attribute would leave it on CPU after `.cuda()`.
- **`coords.unsqueeze(-1) * self.freqs` broadcasting.** `(..., 3, 1) * (F,)` → `(..., 3, F)` computes every axis at every frequency without a Python loop — the vectorization principle hammered in `00_python_into.ipynb`.
- **`flatten(start_dim=-2)`** turns `(..., 3, 2F)` into `(..., 3·2F)` so each axis's sin/cos interleave into one vector; concatenating raw `coords` first preserves the low-frequency signal (helps optimization).
- **`F.scaled_dot_product_attention` over manual softmax.** The notebook wrote `attn = softmax(q @ k.transpose(-2,-1)/sqrt(d_k)); out = attn @ v` to *teach* the mechanism. In production the fused op is numerically safer (it does the scaling and a stable softmax internally) and is dramatically faster on GPU (FlashAttention). It also applies attention dropout for us — note we pass `dropout_p=0.0` at eval (`self.training` guard) so inference is deterministic.
- **`.transpose(1,2).contiguous().view(...)`** — after attention the tensor is `(B, H, Nq, d_k)` with a non-contiguous memory layout from the transpose; `.contiguous()` is required before `.view` can merge the head dims back to `d_model`.
- **Pre-norm (`x + attn(norm(x))`) vs the notebook's post-norm (`norm(x + attn(x))`).** Pre-norm keeps an identity path from input to output of every block, so even a 4–12 layer stack trains without careful warmup gymnastics. This is the one deliberate deviation from 08_GPT and it is the modern default (GPT-2+, ViT).
- **Separate `norm_q` and `norm_kv`.** Query (points) and key/value (image tokens) come from different distributions; giving each its own LayerNorm lets the block calibrate them independently before the dot product.

### Section 5: Verification
```bash
python -m src.model.layers
# Expect: layers.py self-test passed: fourier (2, 128, 63) block (2, 128, 384)
```

## `src/model/encoder.py`

### Section 1: Header
- **Path:** `src/model/encoder.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** Wraps a pretrained **DINOv2 ViT-S/14** and exposes its **patch-token sequence** `(B, 257, 384)` (1 CLS + 256 patch tokens). Provides freeze/unfreeze and optional LoRA. This is the spec's "Step 1: Vision Transformer encoder that splits the image into patches → tokens."

### Section 2: Concept Lesson

**ViT vs CNN — what changes.**
In `04_CNN_Data_Augmentation/4.1 Image Classification CNN.ipynb` a CNN slides small kernels across the image, building a spatial hierarchy and ultimately pooling to a single feature vector — locality and translation-equivariance are baked in. The old `code/` encoder used exactly this (ResNet-18) and collapsed the image to a single 256-d vector. A **Vision Transformer** instead **cuts the image into a grid of fixed patches** (here 14×14 px), linearly embeds each patch into a token, adds positional embeddings, and runs **global self-attention** so every patch can see every other patch from layer 1. The output is not one vector but a **sequence of tokens, one per patch**, each a rich descriptor of its image region. That is precisely what a cross-attention decoder needs: spatially-localized features to attend into.

**Patch tokens.** A 224×224 image with patch size 14 yields a 16×16 = 256 grid of patches → 256 patch tokens, plus 1 special **CLS token** that aggregates global context. We keep both → **257 tokens** of width 384. (DINOv2 also has 4 "register" tokens internally; we use the clean `x_norm_patchtokens` + `x_norm_clstoken` outputs and ignore registers.)

**Why DINOv2 (pretrained) instead of training a ViT from scratch.** Training a ViT from scratch needs enormous labelled data; we have a modest ShapeNet subset. DINOv2 is **self-supervised** on 142M curated images and produces features that are excellent for *dense* tasks (segmentation, depth, correspondence) out of the box — exactly the spatial quality we want for reconstruction. This is the same logic as `05_Transfer_learining_Fine-tuining/5.1 Transfer learning.ipynb`, which loaded ImageNet-pretrained ResNet weights and reused them; we apply the identical "stand on a pretrained backbone" strategy, and the identical **freeze-the-backbone** move (`for p in backbone.parameters(): p.requires_grad = False`, then train only the new head — here, the decoder).

**Borrowed / adapted / new.**
- *Borrowed:* the transfer-learning load-and-freeze pattern (05), the optional LoRA config (`11_LLM_Fine-tuning`, `target_modules=["qkv"]`).
- *Adapted:* instead of replacing a classification head, we **discard the head entirely** and surface the token sequence.
- *New:* `forward_features` token extraction; image-size→token-count assertion (`%14`).

### Section 3: The Complete Code
```python
"""DINOv2 Vision Transformer encoder producing a patch-token sequence.

Applies the pretrained-backbone + freeze strategy from 05_transfer_learning to
a self-supervised ViT, exposing patch tokens (not a single vector) so the
decoder can cross-attend into spatial image features.
"""

from typing import Optional

import torch
import torch.nn as nn

# DINOv2 variant -> patch-token embedding dimension.
_DINOV2_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Encoder(nn.Module):
    """Pretrained DINOv2 ViT exposing ``(B, 1 + n_patches, embed_dim)`` tokens.

    The first token is the CLS token (global summary); the remaining tokens are
    one per image patch. With a 224x224 image and patch size 14 there are
    256 patch tokens, so the sequence length is 257.
    """

    def __init__(
        self,
        variant: str = "dinov2_vits14",
        freeze: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
    ) -> None:
        """Load the backbone from torch.hub and configure trainability.

        Args:
            variant: One of the keys in ``_DINOV2_EMBED_DIM``.
            freeze: If True, freeze all backbone weights (feature-extraction mode).
            use_lora: If True, inject LoRA adapters into the attention qkv layers
                (requires the ``peft`` package). Implies a frozen base.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling alpha.
        """
        super().__init__()
        if variant not in _DINOV2_EMBED_DIM:
            raise ValueError(f"Unknown DINOv2 variant {variant!r}; "
                             f"choose from {list(_DINOV2_EMBED_DIM)}")
        self.variant = variant
        self.embed_dim = _DINOV2_EMBED_DIM[variant]
        self.patch_size = 14

        # Load the self-supervised backbone. trust_repo avoids an interactive prompt.
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", variant, trust_repo=True
        )

        if freeze and not use_lora:
            self.freeze_backbone()

        if use_lora:
            self._apply_lora(lora_r, lora_alpha)

    def freeze_backbone(self) -> None:
        """Disable gradients for every backbone parameter (feature extraction)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _apply_lora(self, r: int, alpha: int) -> None:
        """Wrap the backbone with LoRA adapters on the attention qkv projections.

        Mirrors the LoraConfig from 11_LLM_Fine-tuning, targeting the single
        fused ``qkv`` Linear that DINOv2's attention uses.
        """
        from peft import LoraConfig, get_peft_model  # local import: optional dep

        for param in self.backbone.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv"],
            lora_dropout=0.0,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into a token sequence.

        Args:
            images: ``(B, 3, H, W)`` with ``H == W`` and ``H % 14 == 0``,
                ImageNet-normalized.

        Returns:
            ``(B, 1 + n_patches, embed_dim)`` tokens; index 0 is the CLS token.
        """
        if images.shape[-1] % self.patch_size != 0 or images.shape[-2] % self.patch_size != 0:
            raise ValueError(
                f"Image size {tuple(images.shape[-2:])} must be divisible by "
                f"patch size {self.patch_size}"
            )
        feats = self.backbone.forward_features(images)
        cls = feats["x_norm_clstoken"].unsqueeze(1)        # (B, 1, D)
        patches = feats["x_norm_patchtokens"]              # (B, n_patches, D)
        return torch.cat([cls, patches], dim=1)            # (B, 1 + n_patches, D)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters (trainable by default)."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


if __name__ == "__main__":
    enc = DINOv2Encoder("dinov2_vits14", freeze=True)
    enc.eval()
    with torch.no_grad():
        tokens = enc(torch.randn(2, 3, 224, 224))
    print("encoder.py self-test:", tuple(tokens.shape), "embed_dim", enc.embed_dim)
    assert tokens.shape == (2, 257, 384), tokens.shape
    assert enc.get_num_params(trainable_only=True) == 0  # fully frozen
```

### Section 4: Line-by-Line Walkthrough
- **`torch.hub.load("facebookresearch/dinov2", variant, trust_repo=True)`** downloads and caches the model on first use (`~/.cache/torch/hub`). `trust_repo=True` suppresses the interactive "do you trust this repo?" prompt so scripts run unattended. (Alternative: HuggingFace `transformers` — see `requirements.txt` notes — but hub keeps deps minimal, consistent with the course's `torchvision`/hub usage.)
- **`self.embed_dim` is read from a table, not hardcoded 384.** This is what makes the ViT-B config a one-line change: `dinov2_vitb14` → `embed_dim = 768`, and the decoder's `token_dim` projection adapts automatically.
- **`forward_features(...)` returns a dict.** We assemble CLS + patches ourselves and `cat` on `dim=1` so the CLS sits at index 0. Keeping the CLS gives the decoder a global-context token alongside the 256 local patch tokens (helps when the relevant object spans the whole frame).
- **`freeze_backbone()` also calls `self.backbone.eval()`.** A frozen backbone should not update BatchNorm running stats or apply stochastic depth/dropout — putting it in eval mode freezes those too. (DINOv2 uses LayerNorm, but this is the safe, general habit from 05's transfer-learning section.)
- **The `% patch_size` assertion** catches the most common ViT bug early: feeding a 256×256 or 224×223 image silently produces a different token count and crashes deep inside attention. We fail loudly at the boundary.
- **LoRA import is local** (`from peft import ...` inside the method) so the dependency is only needed if you actually enable LoRA — the default path never imports `peft`.
- **`get_num_params(trainable_only=True) == 0` in the self-test** is the proof that freezing worked: with the backbone frozen and no decoder attached, there is nothing to train here.

### Section 5: Verification
```bash
python -m src.model.encoder
# Expect (first run downloads ~85MB): encoder.py self-test: (2, 257, 384) embed_dim 384
```
> ⚠️ First run downloads DINOv2 weights. If offline, this is the only network dependency in the project — run once while connected; it is then cached.

---

## `src/model/decoder.py`

### Section 1: Header
- **Path:** `src/model/decoder.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The spec's "Step 2." Takes 3D query points `(B, N, 3)` and the encoder's image tokens `(B, Nk, token_dim)` and returns occupancy **logits** `(B, N, 1)`. Each point is encoded with Fourier features, projected to `d_model`, then refined through `n_layers` of cross-attention into the image tokens. Built entirely from `layers.py`.

### Section 2: Concept Lesson

**Cross-attention, mechanically (Q, K, V).**
Attention answers: "for each query, which values matter, and how much?" Each **query** vector is compared (dot product) against every **key** vector to produce similarity scores; `softmax` turns scores into weights that sum to 1; the output is the weighted sum of the **value** vectors. In *self*-attention (08_GPT, 09_BERT) Q, K, V all come from one sequence. In **cross-attention** the queries come from one place and the keys/values from another. Here:
- **Q = the 3D point** (after Fourier encoding + linear projection). "I am the location (x, y, z); which image regions describe me?"
- **K, V = the image patch tokens.** Keys advertise "I am the patch at this image location with these contents"; values carry the actual descriptive features.
So a query point near, say, the chair's leg attends strongly to the patch tokens showing that leg, pulls their features in, and from them decides "inside" vs "outside." This is literally the spec's sentence: *"the 3D points directly interact with the image vectors to understand which parts are important to them."*

**Why we must NOT use self-attention among the query points.**
Self-attention is O(N²) in sequence length. At inference we query a **64³ = 262,144-point grid** (and 2048 points × batch at train time). Self-attention among 262K points would be 262,144² ≈ 6.9×10¹⁰ pairs — impossible. More fundamentally, **occupancy is a point-wise field**: whether (x,y,z) is inside the object does not depend on which *other* query points we happened to sample. Each point should be answerable independently. So the decoder uses **cross-attention only** — points attend to the (small, fixed, 257-token) image, never to each other. This keeps cost **linear** in the number of query points and lets us shovel arbitrary grid batches through the same network.

**The Perceiver-style decoding pattern.**
This independent-queries-cross-attend-into-a-latent-array design is exactly **PerceiverIO** (Jaegle et al. 2021): a large, variable set of output queries each cross-attend into a fixed latent set, with no query-to-query attention. It's the natural fit for "evaluate a continuous field at arbitrarily many coordinates." Our latent set is the DINOv2 token sequence.

**Borrowed / adapted / new.**
- *Borrowed:* every sub-module from `layers.py` (which in turn borrow from 08_GPT / 09_BERT).
- *Adapted:* the GPT/BERT *encoder* stack (self-attention over one sequence) becomes a *decoder* stack of cross-attention blocks over two sequences.
- *New:* Fourier-encoded coordinate queries; a `token_dim → d_model` projection so the decoder is decoupled from the chosen ViT variant; point-wise independence (no self-attention).

### Section 3: The Complete Code
```python
"""Cross-attention occupancy decoder.

Query points (Fourier-encoded) cross-attend into DINOv2 image tokens through a
stack of pre-norm blocks and emit per-point occupancy logits. A PerceiverIO-style
design: queries are independent (no self-attention among points), so cost is
linear in the number of points and a 64^3 grid is tractable.
"""

import torch
import torch.nn as nn

from src.model.layers import CrossAttentionBlock, FourierPositionalEncoding


class CrossAttentionOccupancyDecoder(nn.Module):
    """Map (points, image tokens) -> occupancy logits via cross-attention."""

    def __init__(
        self,
        token_dim: int = 384,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        num_bands: int = 10,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        """Args:
        token_dim: Width of the encoder tokens (DINOv2 embed_dim).
        d_model: Internal decoder width.
        n_heads: Attention heads (must divide d_model).
        n_layers: Number of cross-attention blocks.
        num_bands: Fourier frequency bands for coordinate encoding.
        dropout: Dropout in attention and FFN.
        ffn_mult: FFN hidden expansion factor.
        """
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model

        self.pos_enc = FourierPositionalEncoding(in_dim=3, num_bands=num_bands)
        self.query_proj = nn.Linear(self.pos_enc.output_dim, d_model)
        # Project image tokens to d_model so the decoder is independent of the
        # chosen ViT variant (token_dim 384 for ViT-S, 768 for ViT-B, ...).
        self.token_proj = nn.Linear(token_dim, d_model)

        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads, dropout, ffn_mult)
            for _ in range(n_layers)
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-init linear layers (same init philosophy as the old MLP decoder)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Predict occupancy logits for each query point.

        Args:
            points: ``(B, N, 3)`` query coordinates in ``[-1, 1]``.
            tokens: ``(B, Nk, token_dim)`` image tokens from the encoder.

        Returns:
            ``(B, N, 1)`` raw occupancy logits (apply sigmoid for probabilities).
        """
        # Queries: Fourier-encode coords, then lift to d_model.
        q = self.query_proj(self.pos_enc(points))   # (B, N, d_model)
        # Keys/Values: project tokens once, reuse across all blocks.
        kv = self.token_proj(tokens)                # (B, Nk, d_model)

        for block in self.blocks:
            q = block(q, kv)

        q = self.final_norm(q)
        return self.head(q)                         # (B, N, 1)

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    dec = CrossAttentionOccupancyDecoder(token_dim=384, d_model=384, n_heads=6, n_layers=4)
    pts = torch.rand(2, 2048, 3) * 2 - 1
    tok = torch.randn(2, 257, 384)
    logits = dec(pts, tok)
    print("decoder.py self-test:", tuple(logits.shape), "params", dec.get_num_params())
    assert logits.shape == (2, 2048, 1), logits.shape
```

### Section 4: Line-by-Line Walkthrough
- **`token_proj` is applied once, before the block loop**, and the same `kv` is reused by every block. Projecting inside each block would waste compute and, worse, give each block a *different* view of the image; one shared projection means all blocks attend into one consistent token representation. (Each block still has its own `norm_kv` LayerNorm inside `CrossAttentionBlock`, which is cheap and lets each layer recalibrate.)
- **`query_proj` input dim is `self.pos_enc.output_dim`, never a magic number.** Because `FourierPositionalEncoding` exposes `output_dim` (63 for `num_bands=10`), changing `num_bands` in the config automatically resizes this layer — no edit needed here.
- **`nn.ModuleList(generator)`** registers each block as a submodule so its parameters are tracked and moved by `.to(device)`. A plain Python list would hide them from the optimizer (a classic bug).
- **No sigmoid in `forward`.** We return raw logits because training uses `BCEWithLogitsLoss` (numerically stable log-sum-exp internally — see `train.py`). Sigmoid is applied only at inference, in `occupancy_network.generate_occupancy_grid`. This matches `03_Pytorch_Intro/3.3 Classification.ipynb`, which used `BCEWithLogitsLoss` on raw logits rather than `BCELoss` on sigmoid outputs.
- **Xavier init** mirrors the original `code/` decoder's initialization, keeping activation variance stable through the stack at step 0.
- **Interface guarantee:** input `tokens` last dim must equal `token_dim` (384 from the ViT-S encoder). `test_shapes.py` asserts exactly this boundary.

### Section 5: Verification
```bash
python -m src.model.decoder
# Expect: decoder.py self-test: (2, 2048, 1) params <a positive integer ~ 4.7M>
```

## `src/model/occupancy_network.py`

### Section 1: Header
- **Path:** `src/model/occupancy_network.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The top-level model. Wires `DINOv2Encoder` → `CrossAttentionOccupancyDecoder`, builds from a config object, runs grid inference (the spec's "Step 3" front half), and handles checkpoint save/load. This is the class `train.py`, `inference.py`, and `evaluate.py` all import.

### Section 2: Concept Lesson

**Composition over a monolith.** Just as `occupancy_network.py` in the old `code/` combined `ResNetEncoder` + `OccupancyDecoder`, we compose our two new modules. The training loop should not know whether the encoder is a CNN or a ViT — it calls `model(images, points)`. This separation (the same `nn.Module` composition seen since `02_Neural_Networks_Backpropagation.ipynb`) is what made the migration a *swap* rather than a *rewrite of everything*.

**Encode the image once, reuse tokens across all grid batches.** At inference we evaluate 64³ = 262,144 points. They don't fit in one forward pass, so we chunk them (e.g. 100K at a time). But the **image tokens are identical for every chunk** — the picture doesn't change. So we run the (expensive) ViT **once**, cache the `(1, 257, 384)` tokens, and feed them to the cheap decoder for each point chunk. Re-encoding the image per chunk would multiply ViT cost by `262144/100000 ≈ 3×` for zero benefit. The old `code/` did the analogous thing with its single latent vector; we do it with the token sequence.

**`from_config` factory.** Rather than threading a dozen kwargs, we build the model from the parsed YAML config (`src/utils/config.py`). This keeps a single source of truth (the contract table at the top) and means every script constructs the model identically.

### Section 3: The Complete Code
```python
"""OccupancyNetwork: DINOv2 encoder + cross-attention decoder, with grid inference."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.model.decoder import CrossAttentionOccupancyDecoder
from src.model.encoder import DINOv2Encoder


class OccupancyNetwork(nn.Module):
    """Single-image -> occupancy field model.

    forward(images, points) -> per-point logits.
    generate_occupancy_grid(images, ...) -> dense probability grid for meshing.
    """

    def __init__(
        self,
        encoder_variant: str = "dinov2_vits14",
        freeze_encoder: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        num_bands: int = 10,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        """Construct encoder + decoder. See the interface contract for defaults."""
        super().__init__()
        self.config_snapshot = {
            "encoder_variant": encoder_variant,
            "freeze_encoder": freeze_encoder,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "num_bands": num_bands,
            "dropout": dropout,
            "ffn_mult": ffn_mult,
        }
        self.encoder = DINOv2Encoder(
            variant=encoder_variant,
            freeze=freeze_encoder,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.decoder = CrossAttentionOccupancyDecoder(
            token_dim=self.encoder.embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_bands=num_bands,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )

    @classmethod
    def from_config(cls, config: Any) -> "OccupancyNetwork":
        """Build from a parsed config (see src/utils/config.py).

        Args:
            config: Object with ``.model.encoder.*`` and ``.model.decoder.*`` fields.
        """
        enc = config.model.encoder
        dec = config.model.decoder
        return cls(
            encoder_variant=enc.variant,
            freeze_encoder=enc.freeze,
            use_lora=enc.use_lora,
            lora_r=enc.lora_r,
            lora_alpha=enc.lora_alpha,
            d_model=dec.d_model,
            n_heads=dec.n_heads,
            n_layers=dec.n_layers,
            num_bands=dec.num_bands,
            dropout=dec.dropout,
            ffn_mult=dec.ffn_mult,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Run the ViT once. Args: images ``(B,3,H,W)``. Returns tokens ``(B,Nk,token_dim)``."""
        return self.encoder(images)

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Predict occupancy logits.

        Args:
            images: ``(B, 3, H, W)`` ImageNet-normalized.
            points: ``(B, N, 3)`` coordinates in ``[-1, 1]``.

        Returns:
            ``(B, N, 1)`` occupancy logits.
        """
        tokens = self.encode(images)
        return self.decoder(points, tokens)

    @torch.no_grad()
    def generate_occupancy_grid(
        self,
        image: torch.Tensor,
        resolution: int = 64,
        query_batch_size: int = 100000,
        bounds: tuple = (-1.0, 1.0),
    ) -> np.ndarray:
        """Evaluate the occupancy field on a dense cubic grid.

        Encodes the image once, then sweeps the grid in chunks through the decoder.

        Args:
            image: ``(1, 3, H, W)`` single image (batch size must be 1).
            resolution: Grid side length R; grid has R^3 points.
            query_batch_size: Points per decoder forward pass.
            bounds: (low, high) coordinate range per axis.

        Returns:
            ``(R, R, R)`` numpy array of occupancy probabilities in ``[0, 1]``.
        """
        if image.shape[0] != 1:
            raise ValueError(f"generate_occupancy_grid expects batch size 1, got {image.shape[0]}")
        device = next(self.parameters()).device
        self.eval()

        tokens = self.encode(image.to(device))  # (1, Nk, token_dim) — computed ONCE

        axis = torch.linspace(bounds[0], bounds[1], resolution, device=device)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)

        probs = torch.empty(points.shape[0], device=device)
        for start in range(0, points.shape[0], query_batch_size):
            chunk = points[start:start + query_batch_size].unsqueeze(0)  # (1, b, 3)
            logits = self.decoder(chunk, tokens)                          # (1, b, 1)
            probs[start:start + chunk.shape[1]] = torch.sigmoid(logits).squeeze(0).squeeze(-1)

        return probs.reshape(resolution, resolution, resolution).cpu().numpy()

    def save(self, path: str) -> None:
        """Save weights + config snapshot so the model can be rebuilt exactly."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config_snapshot}, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "OccupancyNetwork":
        """Rebuild a model from a checkpoint produced by ``save``."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters (trainable by default)."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


if __name__ == "__main__":
    model = OccupancyNetwork()
    model.eval()
    img = torch.randn(2, 3, 224, 224)
    pts = torch.rand(2, 512, 3) * 2 - 1
    with torch.no_grad():
        out = model(img, pts)
    print("occupancy_network.py forward:", tuple(out.shape))
    assert out.shape == (2, 512, 1)

    grid = model.generate_occupancy_grid(img[:1], resolution=16, query_batch_size=1024)
    print("grid:", grid.shape, "range", round(float(grid.min()), 3), round(float(grid.max()), 3))
    assert grid.shape == (16, 16, 16)
```

### Section 4: Line-by-Line Walkthrough
- **`@torch.no_grad()` on `generate_occupancy_grid`** disables autograd for the whole method — no graph is built, halving memory and speeding up the 262K-point sweep. Inference never needs gradients.
- **`tokens = self.encode(...)` is hoisted out of the loop.** This is the "encode once, reuse" optimization spelled out in the concept lesson — the single most impactful inference speedup.
- **`torch.meshgrid(..., indexing="ij")`** must use `"ij"` (matrix indexing) so axis order is (x, y, z); the default `"xy"` would swap the first two axes and mirror the mesh. The old `code/` used `"ij"` for the same reason — we preserve it so Marching Cubes orientation stays correct.
- **Pre-allocate `probs = torch.empty(R^3)` and fill slices** instead of `append`+`cat`. With 262K entries this avoids building a Python list of tensors and a final concatenation, reducing peak memory and fragmentation.
- **`sigmoid` is applied here, not in the decoder.** Training wants logits (for `BCEWithLogitsLoss`); inference wants probabilities (for thresholding at 0.5). Centralizing the sigmoid at the single inference entry point keeps the two regimes unambiguous.
- **`config_snapshot` saved alongside weights.** `from_checkpoint` rebuilds the exact architecture (variant, d_model, n_layers, …) before loading weights, so a checkpoint is self-describing — you never have to remember which config produced it. `weights_only=False` is required because we store a Python dict of config values (the old `code/` used the same flag).
- **`from_config` is the only construction path the scripts use**, guaranteeing train/infer/eval build identical models from the same YAML.

### Section 5: Verification
```bash
python -m src.model.occupancy_network
# Expect:
# occupancy_network.py forward: (2, 512, 1)
# grid: (16, 16, 16) range <~0.x> <~0.x>
```
> This is the **SYNC 1** gate for Dev A. Combined with `verify_setup.py`, it proves the full model composes and produces correct shapes end-to-end.

# Data

## `src/data/__init__.py`

### Section 1: Header
- **Path:** `src/data/__init__.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Public API of the data package: re-exports `ShapeNetDataset`, `get_dataloader`, and `CATEGORIES`.

### Section 2: Concept
Same curated-surface idea as `model/__init__.py`: `train.py` does `from src.data import get_dataloader` without caring about file layout.

### Section 3: Code
```python
"""Data package: ShapeNet image+occupancy loading and preprocessing."""

from src.data.dataset import CATEGORIES, ShapeNetDataset, get_dataloader
from src.data.preprocessing import ImagePreprocessor

__all__ = ["CATEGORIES", "ShapeNetDataset", "get_dataloader", "ImagePreprocessor"]
```

### Section 4: Walkthrough
- Re-exports both the dataset and the preprocessor so callers have one import location.

### Section 5: Verification
```bash
python -c "from src.data import get_dataloader, ShapeNetDataset, ImagePreprocessor, CATEGORIES; print(len(CATEGORIES))"
# Expect: 13
```

---

## `src/data/preprocessing.py`

### Section 1: Header
- **Path:** `src/data/preprocessing.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Turns a PIL/numpy/path image into the exact tensor DINOv2 expects: `(3, 224, 224)`, ImageNet-normalized. Provides train-time augmentation and a `denormalize` helper for visualization. Kept from the old `code/` and **verified DINOv2-compatible**.

### Section 2: Concept Lesson

**Why this file barely changes.** DINOv2 was trained on ImageNet-normalized inputs with `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` — the **identical** constants the old ResNet-18 encoder used (and the same ones in `05 Transfer learning`'s `weights.transforms()`). So the existing `ImagePreprocessor` is already correct for the new encoder. The only new requirement is that the image side be a multiple of the ViT patch size (14); 224 = 16×14 ✓, so we add a defensive assertion.

**Train vs eval transforms.** Exactly the lesson from `04_CNN_Data_Augmentation/4.2 Data Augmentation.ipynb`: augment only the training stream (random crop, flip, color jitter) to improve generalization; the val/test/inference stream gets *only* resize + normalize, so evaluation is deterministic and comparable. Mixing augmentation into eval would make metrics noisy and irreproducible.

### Section 3: The Complete Code
```python
"""Image preprocessing for the DINOv2 encoder.

ImageNet normalization (DINOv2's training distribution; identical to the
constants used in 05 Transfer learning). Train-only augmentation follows the
train/eval split taught in 04_CNN_Data_Augmentation/4.2.
"""

from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    """Convert images to normalized ``(3, image_size, image_size)`` tensors."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        patch_size: int = 14,
    ) -> None:
        """Args:
        image_size: Output side length; must be divisible by ``patch_size``.
        mean: Per-channel normalization mean (ImageNet/DINOv2).
        std: Per-channel normalization std (ImageNet/DINOv2).
        patch_size: ViT patch size used to validate ``image_size``.
        """
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by ViT patch_size ({patch_size})"
            )
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.transform_augment = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        augment: bool = False,
    ) -> torch.Tensor:
        """Preprocess one image.

        Args:
            image: A PIL image, HxWxC numpy array, or path string.
            augment: If True, apply training augmentation.

        Returns:
            ``(3, image_size, image_size)`` normalized float tensor.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform_augment(image) if augment else self.transform(image)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert normalization for visualization. Args/Returns: ``(3, H, W)``."""
        mean = torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
        return tensor * std + mean


if __name__ == "__main__":
    pre = ImagePreprocessor(image_size=224)
    dummy = Image.new("RGB", (300, 200), "white")
    out = pre(dummy)
    print("preprocessing.py self-test:", tuple(out.shape))
    assert out.shape == (3, 224, 224)
```

### Section 4: Line-by-Line Walkthrough
- **The `% patch_size` guard at construction** is the new line versus the old `code/` — it stops a mismatched `image_size` from silently producing the wrong token count downstream.
- **`Resize((image_size+32))` then `RandomCrop(image_size)`** gives translation jitter (the crop position varies) — the standard recipe from 04.2; we keep its magnitudes modest because the object must stay framed for reconstruction.
- **ImageNet `mean`/`std`** are *not* arbitrary; they are DINOv2's expected input statistics. Changing them would shift the input distribution away from what the frozen backbone learned and degrade features.
- **`denormalize`** is for sanity-checking that the loader feeds correct images (and for TensorBoard image logging in `train.py`).

### Section 5: Verification
```bash
python -m src.data.preprocessing
# Expect: preprocessing.py self-test: (3, 224, 224)
```

---

## `src/data/dataset.py`

### Section 1: Header
- **Path:** `src/data/dataset.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** The ShapeNet loader. Reads the OccNet/Choy-2016 layout (`points.npz` for occupancy training; `img_choy2016/*.jpg` for views), with the **new** ability to also return surface points + normals from `pointcloud.npz` (for evaluation metrics) and optional camera parameters. Yields `{image, points, occupancy, ...}` batches.

### Section 2: Concept Lesson

**The dataset/DataLoader pattern** is straight from `03_Pytorch_Intro/3.4 Image Classification.ipynb` and `04`: subclass `torch.utils.data.Dataset`, implement `__len__` and `__getitem__` returning tensors, then wrap in a `DataLoader` for batching/shuffling/workers. Nothing exotic.

**The data layout (OccNet / Choy 2016).** Each model directory contains:
- `points.npz` → `points` `(P, 3)` in `[-0.5, 0.5]` (volume samples) and `occupancies` (P bits, **packed** — 8 occupancies per byte). `np.unpackbits` expands them. These are the **(point, inside/outside)** pairs we train on.
- `pointcloud.npz` → `points` `(Q, 3)` and `normals` `(Q, 3)` sampled **on the surface**. Unused by the old loader; we add it because **Chamfer distance, F-score, and Normal Consistency** (see `metrics.py`) compare predicted vs. ground-truth *surfaces*, which need surface points + normals.
- `img_choy2016/*.jpg` → 24 rendered views from different angles, plus `cameras.npz`. We sample one random view per `__getitem__` (the spec: "images shot from different angles").

**Coordinate normalization detail.** OccNet points live in `[-0.5, 0.5]`. Our decoder/grid use `[-1, 1]`. We **multiply points by 2** at load time so ground-truth points and the inference grid share one coordinate frame — otherwise the mesh would come out half-size and offset. (This is the kind of silent convention bug the contract table exists to prevent.)

**Deterministic splits.** Per-category `np.random.RandomState(42)` permutation → 80/10/10 train/val/test, identical to the old loader, so results are reproducible and the test set never leaks into training.

**Borrowed / adapted / new.**
- *Borrowed:* `Dataset`/`DataLoader` skeleton (03.4, 04), the old `code/` split logic and `unpackbits` handling.
- *Adapted:* the `__getitem__` return dict; `[-0.5,0.5] → [-1,1]` scaling made explicit.
- *New:* `pointcloud.npz` + camera loading behind `return_eval` / `load_camera` flags.

### Section 3: The Complete Code
```python
"""ShapeNet dataset (OccNet/Choy-2016 layout) for single-view occupancy learning.

Standard Dataset/DataLoader pattern from 03.4/04. Returns an image, volume
points with binary occupancy (training), and optionally surface points+normals
and camera parameters (evaluation).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import ImagePreprocessor

# The 13 ShapeNet categories named in the project goal (id -> human name).
CATEGORIES = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "vessel",
}

# OccNet points are stored in [-0.5, 0.5]; we train/infer in [-1, 1].
_POINT_SCALE = 2.0


class ShapeNetDataset(Dataset):
    """ShapeNet image + occupancy dataset with optional surface/camera outputs."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        num_points: int = 2048,
        image_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None,
        return_eval: bool = False,
        eval_points: int = 100000,
        load_camera: bool = False,
    ) -> None:
        """Args:
        root: Dataset root containing one subdir per category id.
        split: 'train' | 'val' | 'test'.
        categories: Category ids to include (default: all 13).
        num_points: Volume points sampled per item for training.
        image_size: Image side length (passed to ImagePreprocessor).
        augment: Apply training image augmentation.
        max_samples: Cap total samples (debugging / smoke tests).
        return_eval: Also return surface points+normals from pointcloud.npz.
        eval_points: Surface points to sample when return_eval is True.
        load_camera: Also return camera intrinsics/extrinsics for the chosen view.
        """
        self.root = Path(root)
        self.split = split
        self.categories = categories or list(CATEGORIES.keys())
        self.num_points = num_points
        self.augment = augment
        self.return_eval = return_eval
        self.eval_points = eval_points
        self.load_camera = load_camera
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.samples = self._find_samples(max_samples)
        print(f"[{split}] {len(self.samples)} samples across "
              f"{len(set(s['category'] for s in self.samples))} categories")

    def _find_samples(self, max_samples: Optional[int]) -> List[Dict[str, str]]:
        """Scan the root and apply the deterministic 80/10/10 per-category split."""
        samples: List[Dict[str, str]] = []
        for cat_id in sorted(self.categories):
            cat_dir = self.root / cat_id
            if not cat_dir.exists():
                continue
            model_dirs = sorted(
                d for d in cat_dir.iterdir()
                if d.is_dir() and (d / "points.npz").exists()
            )
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(model_dirs))
            n = len(model_dirs)
            train_end, val_end = int(0.8 * n), int(0.9 * n)
            if self.split == "train":
                selected = indices[:train_end]
            elif self.split == "val":
                selected = indices[train_end:val_end]
            else:
                selected = indices[val_end:]
            for idx in selected:
                samples.append({
                    "category": cat_id,
                    "model_id": model_dirs[idx].name,
                    "dir": str(model_dirs[idx]),
                })
                if max_samples and len(samples) >= max_samples:
                    return samples
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, model_dir: Path) -> Tuple[Image.Image, int]:
        """Load one random rendered view; return (image, view_index)."""
        img_dir = model_dir / "img_choy2016"
        if img_dir.exists():
            views = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if views:
                vi = int(np.random.randint(len(views)))
                return Image.open(str(views[vi])).convert("RGB"), vi
        return Image.new("RGB", (224, 224), "white"), 0

    def _load_points(self, model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and subsample volume points + binary occupancy, scaled to [-1, 1]."""
        data = np.load(str(model_dir / "points.npz"))
        points = data["points"].astype(np.float32) * _POINT_SCALE
        occ = np.unpackbits(data["occupancies"])[: points.shape[0]].astype(np.float32)
        n = points.shape[0]
        choice = np.random.choice(n, size=self.num_points, replace=n < self.num_points)
        return points[choice], occ[choice]

    def _load_pointcloud(self, model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load surface points + normals (for evaluation), scaled to [-1, 1]."""
        pc_path = model_dir / "pointcloud.npz"
        if not pc_path.exists():
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty
        data = np.load(str(pc_path))
        pts = data["points"].astype(np.float32) * _POINT_SCALE
        normals = data["normals"].astype(np.float32)
        m = pts.shape[0]
        k = min(self.eval_points, m)
        choice = np.random.choice(m, size=k, replace=False)
        return pts[choice], normals[choice]

    def _load_camera(self, model_dir: Path, view_idx: int) -> Dict[str, np.ndarray]:
        """Load camera world matrix + intrinsics for a view; empty dict if absent."""
        cam_path = model_dir / "img_choy2016" / "cameras.npz"
        if not cam_path.exists():
            return {}
        cam = np.load(str(cam_path))
        return {
            "world_mat": cam[f"world_mat_{view_idx}"].astype(np.float32),
            "camera_mat": cam[f"camera_mat_{view_idx}"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one sample dict (see module docstring for keys)."""
        info = self.samples[idx]
        model_dir = Path(info["dir"])

        image, view_idx = self._load_image(model_dir)
        image_tensor = self.preprocessor(image, augment=self.augment)
        points, occupancy = self._load_points(model_dir)

        item: Dict[str, object] = {
            "image": image_tensor,
            "points": torch.from_numpy(points),
            "occupancy": torch.from_numpy(occupancy).unsqueeze(-1),
            "category": info["category"],
            "model_id": info["model_id"],
        }
        if self.return_eval:
            pc_pts, pc_normals = self._load_pointcloud(model_dir)
            item["eval_points"] = torch.from_numpy(pc_pts)
            item["eval_normals"] = torch.from_numpy(pc_normals)
        if self.load_camera:
            cam = self._load_camera(model_dir, view_idx)
            for key, value in cam.items():
                item[key] = torch.from_numpy(value)
        return item


def get_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs: object,
) -> DataLoader:
    """Build a DataLoader over ShapeNetDataset.

    Shuffles + drops the last partial batch only for training. ``**kwargs`` are
    forwarded to ``ShapeNetDataset`` (num_points, augment, return_eval, ...).
    """
    dataset = ShapeNetDataset(root=root, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        loader = get_dataloader(sys.argv[1], split="train", batch_size=4, num_workers=0,
                                max_samples=8)
        batch = next(iter(loader))
        print("image", tuple(batch["image"].shape),
              "points", tuple(batch["points"].shape),
              "occ", tuple(batch["occupancy"].shape))
        assert batch["image"].shape[1:] == (3, 224, 224)
        assert batch["points"].shape[-1] == 3
        assert batch["occupancy"].shape[-1] == 1
        print("dataset.py self-test passed")
    else:
        print("Pass a ShapeNet root to self-test: python -m src.data.dataset <DATA_ROOT>")
```

### Section 4: Line-by-Line Walkthrough
- **`* _POINT_SCALE` (= 2.0) on every loaded point set.** OccNet stores points in `[-0.5, 0.5]`; the decoder, the inference grid, and Marching Cubes all assume `[-1, 1]`. Scaling at the single load point keeps one coordinate frame everywhere. Both `_load_points` and `_load_pointcloud` apply it so training labels and eval surfaces stay aligned.
- **`np.unpackbits(data["occupancies"])[:P]`** — occupancies are bit-packed (8 per byte) to save space; unpacking yields 8×bytes bits, so we slice back to exactly `P` to drop padding bits. (Identical to the old loader.)
- **`np.random.choice(..., replace=n < num_points)`** subsamples to a fixed `num_points` so every item has the same shape for batching, allowing replacement only if the file has fewer points than requested.
- **`RandomState(42)` is created per category, inside the loop.** This makes the split independent of category order/count and reproducible run-to-run — the test set is stable.
- **`return_eval`/`load_camera` are opt-in** and default off, so the training path stays light (no `pointcloud.npz` I/O per step). `evaluate.py` flips `return_eval=True`.
- **Empty-array fallbacks** (`pointcloud.npz` missing) return `(0,3)` tensors rather than crashing — some ShapeNet subsets ship without surface clouds, and eval can skip those metrics gracefully.
- **`drop_last=(split=='train')`** avoids a ragged final training batch (which can destabilize BatchNorm-style stats and AMP); val/test keep all samples for honest metrics.
- **Interface guarantee:** `image (B,3,224,224)`, `points (B,num_points,3)`, `occupancy (B,num_points,1)` — exactly what `train.py` feeds `model(images, points)` and `criterion(pred, occupancy)`.

### Section 5: Verification
```bash
# Without data (no root): prints usage and exits cleanly.
python -m src.data.dataset
# Expect: Pass a ShapeNet root to self-test: ...

# With data:
python -m src.data.dataset /path/to/ShapeNet
# Expect: [train] N samples ... ; image (4, 3, 224, 224) points (4, 2048, 3) occ (4, 2048, 1)
#         dataset.py self-test passed
```

# Mesh

## `src/mesh/__init__.py`

### Section 1: Header
- **Path:** `src/mesh/__init__.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Public API of the mesh package: `extract_mesh`, `MarchingCubesExtractor`, `make_printable`, `save_mesh`, `MeshExporter`.

### Section 2: Concept
Curated package surface so `inference.py` writes `from src.mesh import extract_mesh, make_printable, save_mesh`.

### Section 3: Code
```python
"""Mesh package: occupancy grid -> printable STL."""

from src.mesh.marching_cubes import MarchingCubesExtractor, extract_mesh
from src.mesh.postprocess import make_printable
from src.mesh.export import MeshExporter, save_mesh

__all__ = ["MarchingCubesExtractor", "extract_mesh", "make_printable",
           "MeshExporter", "save_mesh"]
```

### Section 4: Walkthrough
- Import order matters for clarity only (no cycles): extraction → postprocess → export, the order the inference pipeline uses them.

### Section 5: Verification
```bash
python -c "from src.mesh import extract_mesh, make_printable, save_mesh; print('ok')"
# Expect: ok
```

---

## `src/mesh/export.py`

### Section 1: Header
- **Path:** `src/mesh/export.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Write a `trimesh.Trimesh` to disk as STL (binary or ASCII), OBJ, or PLY. Kept from the old `code/`. STL is the spec's required output (3D-printable).

### Section 2: Concept Lesson
**Mesh file formats — what to use when.**
- **STL** (STereoLithography): the lingua franca of 3D printing. Stores only triangles (vertices + face normals), no color/topology metadata. **Binary STL** is compact and fast; **ASCII STL** is human-readable but large. The spec asks for STL → this is our primary output.
- **OBJ**: text format, widely supported by 3D tools, stores vertices/faces/optionally normals and texture coords. Useful for inspecting results in Blender.
- **PLY**: stores per-vertex attributes (e.g., colors, normals) — handy for debugging (e.g., coloring by predicted probability).
`trimesh.export` handles all three; we just route by extension. No notebook covers meshes (the course is 2D/NLP/RL), so this module is **new domain code**, but it's thin I/O.

### Section 3: The Complete Code
```python
"""Export trimesh meshes to STL/OBJ/PLY. STL is the 3D-printing target."""

from pathlib import Path
from typing import Optional

import trimesh


class MeshExporter:
    """Write meshes to disk, creating parent directories as needed."""

    def __init__(self, create_dirs: bool = True) -> None:
        """Args: create_dirs: mkdir -p the output's parent before writing."""
        self.create_dirs = create_dirs

    def _prepare_path(self, path: str) -> Path:
        """Resolve the output path and ensure its directory exists."""
        p = Path(path)
        if self.create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def export_stl(self, mesh: trimesh.Trimesh, path: str, binary: bool = True) -> bool:
        """Export to STL. Args: binary: binary (True) vs ASCII (False). Returns success."""
        try:
            out = self._prepare_path(path)
            mesh.export(str(out), file_type="stl" if binary else "stl_ascii")
            print(f"Exported STL: {out} | V={len(mesh.vertices)} F={len(mesh.faces)} "
                  f"watertight={mesh.is_watertight}")
            return True
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"Failed to export STL: {exc}")
            return False

    def export_obj(self, mesh: trimesh.Trimesh, path: str) -> bool:
        """Export to OBJ. Returns success."""
        try:
            mesh.export(str(self._prepare_path(path)), file_type="obj")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to export OBJ: {exc}")
            return False

    def export_ply(self, mesh: trimesh.Trimesh, path: str) -> bool:
        """Export to PLY. Returns success."""
        try:
            mesh.export(str(self._prepare_path(path)), file_type="ply")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to export PLY: {exc}")
            return False


def save_mesh(mesh: trimesh.Trimesh, path: str, file_format: Optional[str] = None) -> bool:
    """Save a mesh, choosing the format from ``file_format`` or the path extension."""
    exporter = MeshExporter()
    if file_format is None:
        ext = Path(path).suffix.lower()
        file_format = ext[1:] if ext else "stl"
    if file_format == "stl":
        return exporter.export_stl(mesh, path)
    if file_format == "obj":
        return exporter.export_obj(mesh, path)
    if file_format == "ply":
        return exporter.export_ply(mesh, path)
    print(f"Unknown format: {file_format}")
    return False


if __name__ == "__main__":
    import tempfile
    box = trimesh.creation.box(extents=(1, 1, 1))
    with tempfile.TemporaryDirectory() as d:
        ok = save_mesh(box, str(Path(d) / "box.stl"))
    print("export.py self-test:", ok)
    assert ok
```

### Section 4: Line-by-Line Walkthrough
- **`file_type="stl"` vs `"stl_ascii"`** — trimesh defaults to binary STL; we expose ASCII for debugging but default to binary (smaller, the printer-friendly choice).
- **Broad `except` that logs and returns `False`** — export is the last pipeline step; we never want a disk hiccup to crash a long inference run silently. The boolean return lets callers branch.
- **`is_watertight` printed at export** gives the operator immediate feedback on printability (a non-watertight STL may fail to slice).
- **Self-test uses `trimesh.creation.box`** so it needs no model or data — pure I/O check.

### Section 5: Verification
```bash
python -m src.mesh.export
# Expect: Exported STL: .../box.stl | V=8 F=12 watertight=True
#         export.py self-test: True
```

---

## `src/mesh/postprocess.py`

### Section 1: Header
- **Path:** `src/mesh/postprocess.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** **New file.** Turn a raw Marching-Cubes mesh into something **3D-printable**: keep the largest connected component, fill small holes, fix normal orientation, and report watertightness. Directly satisfies the spec's requirement that the STL be printable.

### Section 2: Concept Lesson

**Why raw Marching Cubes output is often not printable.**
A 3D printer's slicer needs a **watertight, manifold** mesh — a closed surface with a well-defined inside and outside. Marching Cubes on a probability grid can produce:
1. **Open surfaces** where the object touches the grid boundary (fixed by zero-padding the grid — see `marching_cubes.py`).
2. **Disconnected floaters** — small spurious components from noisy probabilities.
3. **Holes** where the iso-surface is ambiguous.
4. **Inconsistent normals** (some faces wound inside-out).

`make_printable` addresses 2–4 with trimesh: `split()` → keep largest component; `fill_holes()`; `fix_normals()`; then check `is_watertight`. This is new domain code (no course notebook covers meshing), but each operation is a one-call trimesh utility.

### Section 3: The Complete Code
```python
"""Post-process a raw mesh into a watertight, printable solid."""

from typing import Optional

import numpy as np
import trimesh


def make_printable(
    mesh: trimesh.Trimesh,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> Optional[trimesh.Trimesh]:
    """Clean a Marching-Cubes mesh for 3D printing.

    Steps: drop degenerate faces, optionally keep only the largest connected
    component, fill holes, and fix normal orientation.

    Args:
        mesh: Raw mesh from Marching Cubes (may be None upstream).
        keep_largest: Discard all but the largest connected component.
        fill_holes: Attempt to close small holes.

    Returns:
        Cleaned mesh, or None if the input was None/empty.
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    if keep_largest:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))

    if fill_holes:
        mesh.fill_holes()

    mesh.fix_normals()

    print(f"make_printable: V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} volume={mesh.volume:.4f}")
    return mesh


if __name__ == "__main__":
    # Two disjoint boxes -> keep_largest should return a single box.
    big = trimesh.creation.box(extents=(2, 2, 2))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation([5, 0, 0])
    combined = trimesh.util.concatenate([big, small])
    cleaned = make_printable(combined)
    print("postprocess.py self-test components after clean:",
          len(cleaned.split(only_watertight=False)))
    assert len(cleaned.split(only_watertight=False)) == 1
    assert cleaned.is_watertight
```

### Section 4: Line-by-Line Walkthrough
- **Order matters:** remove degenerate/duplicate faces *first* (so component splitting and hole filling operate on clean topology), then split, then fill, then fix normals (normals must be recomputed after topology edits).
- **`split(only_watertight=False)` + `max(..., key=len(faces))`** keeps the dominant blob and deletes floaters; `only_watertight=False` is required because the components aren't watertight *yet* (we fill holes afterward).
- **`fix_normals()` last** ensures consistent outward winding so the slicer (and `is_watertight`) interpret inside/outside correctly.
- **Guard for `None`/empty** lets the inference script handle "no surface found" without a crash (Marching Cubes returns `None` when the grid never crosses the threshold).
- **Self-test with two disjoint boxes** proves component selection works without needing the model.

### Section 5: Verification
```bash
python -m src.mesh.postprocess
# Expect: make_printable: V=8 F=12 watertight=True volume=8.0000
#         postprocess.py self-test components after clean: 1
```

---

## `src/mesh/marching_cubes.py`

### Section 1: Header
- **Path:** `src/mesh/marching_cubes.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Convert a `(R,R,R)` probability grid into a triangle mesh via `skimage.measure.marching_cubes`, rescaling vertices to `[-1,1]³`. **Extended** with optional **zero-border padding** so surfaces that touch the grid edge close into a watertight solid. Optionally chains into `make_printable`.

### Section 2: Concept Lesson

**Marching Cubes in one paragraph.** Given a scalar field sampled on a grid and an iso-level (here probability = 0.5), Marching Cubes walks every cell of the grid, looks at which of the 8 corners are above/below the level, and emits triangles approximating where the surface crosses that cell. The result is a triangle mesh of the level-set surface. `skimage` implements the classic Lorensen–Cline algorithm. The spec names this exact algorithm in Step 3.

**Why zero-border padding.** If the object's occupancy is still high at the edge of the `[-1,1]` cube (e.g., a table top reaching the boundary), the surface never "comes back down" to 0 inside the grid, so Marching Cubes leaves an **open hole** at that face — not printable. Padding the grid with a one-voxel border of zeros forces the field to cross 0.5 just inside the boundary, **closing the surface**. It's a one-line `np.pad` that eliminates the most common open-mesh artifact.

This is new domain code (no notebook covers it), adapted from the old `code/`'s extractor with padding + a postprocess hook added.

### Section 3: The Complete Code
```python
"""Marching Cubes: probability grid -> triangle mesh, with edge-closing padding."""

from typing import Optional, Tuple

import numpy as np
import trimesh
from skimage import measure

from src.mesh.postprocess import make_printable


class MarchingCubesExtractor:
    """Extract an iso-surface mesh from an occupancy probability grid."""

    def __init__(self, threshold: float = 0.5, pad: bool = True) -> None:
        """Args:
        threshold: Iso-level (probability) defining the surface.
        pad: Zero-pad the grid by 1 voxel so edge-touching surfaces close.
        """
        self.threshold = threshold
        self.pad = pad

    def extract(
        self,
        occupancy_grid: np.ndarray,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        postprocess: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        """Run Marching Cubes and map vertices into world bounds.

        Args:
            occupancy_grid: ``(R, R, R)`` probabilities in [0, 1].
            bounds: (low, high) world coordinate range the grid spans.
            postprocess: If True, run ``make_printable`` on the result.

        Returns:
            A ``trimesh.Trimesh`` or None if no surface crosses the threshold.
        """
        grid = occupancy_grid
        if self.pad:
            grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0.0)

        if grid.max() < self.threshold or grid.min() > self.threshold:
            print("Warning: no surface at this threshold (grid never crosses level)")
            return None

        try:
            verts, faces, normals, _ = measure.marching_cubes(grid, level=self.threshold)
        except (ValueError, RuntimeError) as exc:
            print(f"Marching cubes failed: {exc}")
            return None

        # Map voxel indices -> [0,1] -> world bounds. Account for the padding
        # offset so geometry stays aligned with the original [-1,1] frame.
        res = grid.shape[0]
        if self.pad:
            verts = verts - 1.0   # undo the 1-voxel pad shift
            res = res - 2         # original resolution
        verts = verts / max(res - 1, 1)
        verts = verts * (bounds[1] - bounds[0]) + bounds[0]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.fix_normals()

        if postprocess:
            mesh = make_printable(mesh)
        return mesh


def extract_mesh(
    occupancy_grid: np.ndarray,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    pad: bool = True,
    postprocess: bool = True,
) -> Optional[trimesh.Trimesh]:
    """Convenience wrapper around ``MarchingCubesExtractor``."""
    return MarchingCubesExtractor(threshold=threshold, pad=pad).extract(
        occupancy_grid, bounds=bounds, postprocess=postprocess
    )


if __name__ == "__main__":
    # A solid sphere occupancy field reaching the grid edge: padding must close it.
    R = 48
    ax = np.linspace(-1, 1, R)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sphere = (np.sqrt(gx**2 + gy**2 + gz**2) < 0.9).astype(np.float32)
    mesh = extract_mesh(sphere, threshold=0.5)
    print("marching_cubes.py self-test:",
          "V", len(mesh.vertices), "F", len(mesh.faces), "watertight", mesh.is_watertight)
    assert mesh is not None and mesh.is_watertight
```

### Section 4: Line-by-Line Walkthrough
- **`np.pad(..., constant_values=0.0)`** adds the zero border; the subsequent `verts - 1.0` and `res - 2` exactly undo the index shift so the mesh lands in the same `[-1,1]` frame as the ground truth (and the dataset's `_POINT_SCALE`). Getting this offset wrong would shrink/shift the mesh — hence the explicit correction.
- **`grid.max() < threshold or grid.min() > threshold` early return** detects "all empty" or "all full" grids (no level crossing) and returns `None` instead of letting skimage raise — the caller (`inference.py`) handles `None` by suggesting a different threshold.
- **`vertex_normals=normals` from skimage, then `fix_normals()`** — skimage gives gradient normals; trimesh's `fix_normals` enforces consistent outward orientation for printing.
- **`postprocess=True` chains `make_printable`** so the default `extract_mesh(...)` returns a print-ready mesh in one call, but eval can pass `postprocess=False` to measure the raw surface.
- **Self-test uses a sphere reaching radius 0.9** (close to the edge) to prove padding closes the surface into a watertight solid.

### Section 5: Verification
```bash
python -m src.mesh.marching_cubes
# Expect: make_printable: ... watertight=True ...
#         marching_cubes.py self-test: V <n> F <n> watertight True
```

# Evaluation

## `src/eval/__init__.py`

### Section 1: Header
- **Path:** `src/eval/__init__.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Public API of the eval package: the four metric functions.

### Section 2: Concept
Curated surface so `evaluate.py` (and tests) import `from src.eval.metrics import ...`. We re-export the metrics for convenience.

### Section 3: Code
```python
"""Evaluation package: reconstruction quality metrics."""

from src.eval.metrics import (
    volumetric_iou,
    chamfer_l1,
    normal_consistency,
    f_score,
)

__all__ = ["volumetric_iou", "chamfer_l1", "normal_consistency", "f_score"]
```

### Section 4: Walkthrough
- Pure re-export; no logic.

### Section 5: Verification
```bash
python -c "from src.eval import volumetric_iou, chamfer_l1, normal_consistency, f_score; print('ok')"
# Expect: ok
```

---

## `src/eval/metrics.py`

### Section 1: Header
- **Path:** `src/eval/metrics.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** **New file.** The four standard single-view reconstruction metrics the spec omitted: **Volumetric IoU** (occupancy agreement), **Chamfer-L1** (surface distance), **Normal Consistency** (surface orientation agreement), **F-Score@τ** (precision/recall of surface points). These turn "loss went down" into "the shapes are actually correct."

### Section 2: Concept Lesson

**Why BCE loss is not enough.** Cross-entropy on sampled points tells you the classifier is calibrated, not whether the *reconstructed shape* matches. Two models with similar BCE can produce very different meshes. The field (ONet, 3D-RETR, Pixel2Mesh) evaluates with geometry-aware metrics:

- **Volumetric IoU** — over a set of test points, `IoU = |pred∩gt| / |pred∪gt|` where membership is `prob > 0.5` vs ground-truth occupancy. Range [0,1], higher better. This is the headline number in the ONet paper. (Conceptually the same intersection-over-union idea you'd use for segmentation; here on 3D occupancy.)
- **Chamfer-L1 distance** — sample points on the predicted surface and on the GT surface; for each predicted point find its nearest GT point and vice-versa; average both directions. Measures geometric closeness; lower better. We use a KD-tree (`scipy.spatial.cKDTree`) for fast nearest-neighbor.
- **Normal Consistency** — like Chamfer but compares **surface normals**: for each nearest pair, take `|⟨n_pred, n_gt⟩|` (absolute cosine), average both directions. Captures whether surfaces face the same way (sharpness/orientation), not just position. Range [0,1], higher better.
- **F-Score@τ** — `precision` = fraction of predicted points within distance τ of GT; `recall` = fraction of GT points within τ of predicted; `F = 2PR/(P+R)`. Robust, threshold-based; higher better. (Tatarchenko et al. argue F-score is the most informative single number.)

These are **new** (no notebook covers 3D metrics), but each reduces to nearest-neighbor queries + simple arithmetic. We accept numpy arrays so they're framework-agnostic and unit-testable without a GPU.

### Section 3: The Complete Code
```python
"""Reconstruction quality metrics: IoU, Chamfer-L1, Normal Consistency, F-Score.

Standard single-view reconstruction metrics (ONet / 3D-RETR). All functions take
numpy arrays so they are easy to unit-test without a GPU.
"""

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def volumetric_iou(
    pred_prob: np.ndarray,
    gt_occ: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Intersection-over-union of occupancy over a shared set of points.

    Args:
        pred_prob: ``(N,)`` predicted occupancy probabilities in [0, 1].
        gt_occ: ``(N,)`` ground-truth occupancy in {0, 1}.
        threshold: Probability above which a point is predicted occupied.

    Returns:
        IoU in [0, 1] (1.0 if both sets are empty).
    """
    pred = pred_prob > threshold
    gt = gt_occ > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _nn_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each point in ``a`` return distance to and index of nearest in ``b``."""
    tree = cKDTree(b)
    dist, idx = tree.query(a, k=1)
    return dist, idx


def chamfer_l1(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two point sets.

    Args:
        pred_points: ``(P, 3)`` points sampled on the predicted surface.
        gt_points: ``(Q, 3)`` points sampled on the ground-truth surface.

    Returns:
        Mean of both directional nearest distances (lower is better);
        ``inf`` if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    d_pred_to_gt, _ = _nn_distances(pred_points, gt_points)
    d_gt_to_pred, _ = _nn_distances(gt_points, pred_points)
    return float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))


def normal_consistency(
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
    gt_points: np.ndarray,
    gt_normals: np.ndarray,
) -> float:
    """Symmetric absolute-cosine agreement of normals at nearest neighbors.

    Args:
        pred_points/pred_normals: ``(P, 3)`` predicted surface points and normals.
        gt_points/gt_normals: ``(Q, 3)`` ground-truth surface points and normals.

    Returns:
        Mean absolute cosine similarity in [0, 1] (higher is better);
        0.0 if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    def _unit(n: np.ndarray) -> np.ndarray:
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-10)

    pn, gn = _unit(pred_normals), _unit(gt_normals)
    _, idx_p2g = _nn_distances(pred_points, gt_points)
    _, idx_g2p = _nn_distances(gt_points, pred_points)
    cos_p2g = np.abs(np.sum(pn * gn[idx_p2g], axis=1))
    cos_g2p = np.abs(np.sum(gn * pn[idx_g2p], axis=1))
    return float(0.5 * (cos_p2g.mean() + cos_g2p.mean()))


def f_score(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    tau: float = 0.02,
) -> float:
    """F-Score: harmonic mean of precision and recall at distance threshold ``tau``.

    Args:
        pred_points: ``(P, 3)`` predicted surface points.
        gt_points: ``(Q, 3)`` ground-truth surface points.
        tau: Distance threshold counting a point as "matched".

    Returns:
        F-score in [0, 1] (higher is better); 0.0 if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0
    d_pred_to_gt, _ = _nn_distances(pred_points, gt_points)
    d_gt_to_pred, _ = _nn_distances(gt_points, pred_points)
    precision = float((d_pred_to_gt < tau).mean())
    recall = float((d_gt_to_pred < tau).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # Identical point sets => perfect scores / zero distance.
    pts = rng.rand(500, 3).astype(np.float32)
    normals = rng.randn(500, 3).astype(np.float32)
    assert chamfer_l1(pts, pts) == 0.0
    assert abs(normal_consistency(pts, normals, pts, normals) - 1.0) < 1e-5
    assert f_score(pts, pts, tau=0.01) == 1.0
    # IoU sanity
    pred = np.array([0.9, 0.9, 0.1, 0.1])
    gt = np.array([1.0, 0.0, 0.0, 1.0])
    iou = volumetric_iou(pred, gt)  # pred occ {0,1}; gt occ {0,3} -> inter 1, union 3
    print("metrics.py self-test: chamfer 0.0, NC 1.0, F 1.0, IoU", round(iou, 3))
    assert abs(iou - (1 / 3)) < 1e-6
```

### Section 4: Line-by-Line Walkthrough
- **Everything is numpy + `cKDTree`** — no torch, no GPU. Metrics are pure functions of arrays, so `test_shapes.py` can check them with tiny synthetic inputs and they run in CI.
- **`cKDTree.query(a, k=1)`** is O(N log N) nearest-neighbor; building one tree per direction and querying is far faster than the O(PQ) brute-force pairwise distance matrix (which for 100K×100K points is impossible).
- **Symmetric (both-direction) averaging** in Chamfer/NC/F-score is the standard definition — a one-directional distance can be fooled (predict one perfectly-placed point → great precision, terrible recall). Both directions penalize both over- and under-coverage.
- **`+ 1e-10` in `_unit`** avoids divide-by-zero for degenerate (zero-length) normals.
- **`np.abs` on the normal cosine** — surfaces can be represented with either normal direction (±n describe the same plane), so we compare orientation, not signed direction.
- **IoU returns 1.0 for empty∪empty** — a degenerate but well-defined case (predicting "all outside" for an all-outside region is perfect agreement).
- **Self-test asserts the analytic answers** (identical sets → Chamfer 0, NC 1, F 1; the hand-computed IoU = 1/3), so a regression in any metric fails loudly.

### Section 5: Verification
```bash
python -m src.eval.metrics
# Expect: metrics.py self-test: chamfer 0.0, NC 1.0, F 1.0, IoU 0.333
```

---

## `src/eval/evaluate.py`

### Section 1: Header
- **Path:** `src/eval/evaluate.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** **New file.** The evaluation driver: load a checkpoint, run over the test split, and report per-category and mean IoU / Chamfer-L1 / Normal Consistency / F-Score. Ties together `OccupancyNetwork` (Dev A), `ShapeNetDataset` (Dev B), `metrics.py`, and the mesh extractor.

### Section 2: Concept Lesson

**An evaluation loop is a no-grad inference loop plus bookkeeping.** Same `model.eval()` + `torch.no_grad()` skeleton as the `test_step` from `03.4`/`06.3`, but instead of accuracy we accumulate the geometry metrics. For each test model we: (1) compute IoU on the dataset's volume points (cheap — one decoder pass on the GT points); (2) generate the occupancy grid, mesh it, sample points from the predicted mesh, and compare to the GT surface cloud for Chamfer/NC/F-score. We aggregate per category (so we can see "cars good, lamps hard") and overall.

This is the **SYNC 3** integration artifact for Dev B — it exercises nearly every other file.

### Section 3: The Complete Code
```python
"""Evaluation driver: checkpoint -> test-split reconstruction metrics."""

import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import trimesh

from src.data import CATEGORIES, get_dataloader
from src.eval.metrics import chamfer_l1, f_score, normal_consistency, volumetric_iou
from src.mesh import extract_mesh
from src.model import OccupancyNetwork
from src.utils.config import load_config


def _sample_mesh_surface(mesh: trimesh.Trimesh, n: int) -> tuple:
    """Sample ``n`` points + their face normals from a mesh surface."""
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32)


@torch.no_grad()
def evaluate(
    model: OccupancyNetwork,
    data_root: str,
    device: torch.device,
    resolution: int = 64,
    threshold: float = 0.5,
    max_samples: int = None,
    surface_samples: int = 50000,
    f_tau: float = 0.02,
) -> Dict[str, Dict[str, float]]:
    """Run reconstruction metrics over the test split.

    Returns a dict mapping category name (and 'mean') to a dict of metric values.
    """
    model.eval()
    loader = get_dataloader(
        root=data_root, split="test", batch_size=1, num_workers=0,
        max_samples=max_samples, return_eval=True, augment=False,
    )
    per_cat: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for batch in loader:
        image = batch["image"].to(device)             # (1, 3, 224, 224)
        points = batch["points"].to(device)           # (1, N, 3)
        gt_occ = batch["occupancy"].squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
        cat = CATEGORIES.get(batch["category"][0], batch["category"][0])

        # --- IoU on the dataset's volume points ---
        logits = model(image, points)                 # (1, N, 1)
        pred_prob = torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu().numpy()
        iou = volumetric_iou(pred_prob, gt_occ, threshold)

        # --- Surface metrics via meshing ---
        grid = model.generate_occupancy_grid(image, resolution=resolution)
        mesh = extract_mesh(grid, threshold=threshold, postprocess=True)

        gt_pts = batch["eval_points"].squeeze(0).cpu().numpy()
        gt_normals = batch["eval_normals"].squeeze(0).cpu().numpy()

        if mesh is not None and len(mesh.faces) > 0 and len(gt_pts) > 0:
            pred_pts, pred_normals = _sample_mesh_surface(mesh, surface_samples)
            cd = chamfer_l1(pred_pts, gt_pts)
            nc = normal_consistency(pred_pts, pred_normals, gt_pts, gt_normals)
            fs = f_score(pred_pts, gt_pts, tau=f_tau)
        else:
            cd, nc, fs = float("inf"), 0.0, 0.0

        per_cat[cat].append({"iou": iou, "chamfer": cd, "nc": nc, "fscore": fs})

    # Aggregate (ignore inf chamfer when averaging).
    results: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict[str, float]] = []
    for cat, rows in per_cat.items():
        results[cat] = _aggregate(rows)
        all_rows.extend(rows)
    results["mean"] = _aggregate(all_rows)
    return results


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean each metric across rows, skipping non-finite Chamfer values."""
    if not rows:
        return {"iou": 0.0, "chamfer": float("inf"), "nc": 0.0, "fscore": 0.0, "n": 0}
    finite_cd = [r["chamfer"] for r in rows if np.isfinite(r["chamfer"])]
    return {
        "iou": float(np.mean([r["iou"] for r in rows])),
        "chamfer": float(np.mean(finite_cd)) if finite_cd else float("inf"),
        "nc": float(np.mean([r["nc"] for r in rows])),
        "fscore": float(np.mean([r["fscore"] for r in rows])),
        "n": len(rows),
    }


def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the metric table."""
    header = f"{'category':<12}{'n':>5}{'IoU':>9}{'Chamfer':>11}{'NC':>9}{'F-score':>9}"
    print(header)
    print("-" * len(header))
    for cat in sorted(k for k in results if k != "mean"):
        r = results[cat]
        print(f"{cat:<12}{r['n']:>5}{r['iou']:>9.4f}{r['chamfer']:>11.4f}"
              f"{r['nc']:>9.4f}{r['fscore']:>9.4f}")
    r = results["mean"]
    print("-" * len(header))
    print(f"{'MEAN':<12}{r['n']:>5}{r['iou']:>9.4f}{r['chamfer']:>11.4f}"
          f"{r['nc']:>9.4f}{r['fscore']:>9.4f}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a 3DScan checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OccupancyNetwork.from_checkpoint(args.checkpoint, map_location=str(device)).to(device)

    resolution = args.resolution or config.inference.grid_resolution
    threshold = args.threshold or config.inference.threshold

    results = evaluate(
        model, args.data_root, device,
        resolution=resolution, threshold=threshold, max_samples=args.max_samples,
    )
    _print_table(results)


if __name__ == "__main__":
    main()
```

### Section 4: Line-by-Line Walkthrough
- **`batch_size=1, num_workers=0`** — evaluation meshes one model at a time (Marching Cubes + trimesh sampling are CPU-bound and per-sample), so batching buys nothing and complicates the grid call (which requires batch size 1).
- **Two metric tiers:** IoU uses the dataset's volume points (one cheap decoder pass); the surface metrics require the full grid→mesh→sample pipeline. Splitting them means even if meshing fails (`mesh is None`), we still report IoU.
- **`_aggregate` skips non-finite Chamfer** so one failed mesh (inf distance) doesn't poison the category mean; `n` reports the sample count for honesty.
- **`trimesh.sample.sample_surface` + `face_normals[face_idx]`** gives predicted surface points *with* normals for Normal Consistency — the same surface-sampling idea the old `code/`'s `PointSampler` used.
- **`postprocess=True` when meshing for eval** — we evaluate the *printable* mesh (the actual deliverable), so metrics reflect what the user receives. (You could pass `postprocess=False` to score the raw iso-surface; we deliberately score the real output.)
- **`from_checkpoint` rebuilds the exact architecture** from the saved config snapshot, so eval never needs to be told which variant/dims the checkpoint used.

### Section 5: Verification
```bash
# Needs a trained checkpoint + data. Smoke form:
python src/eval/evaluate.py --checkpoint checkpoints/best.pt --data-root <DATA> --max-samples 16
# Expect: a table with per-category rows and a MEAN row; IoU in [0,1], Chamfer small positive,
#         NC in [0,1], F-score in [0,1].
```

# Config

## `src/utils/__init__.py`

### Section 1: Header
- **Path:** `src/utils/__init__.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Public API of utils: `load_config`, `Config`.

### Section 2: Concept
Same curated-surface pattern; lets scripts write `from src.utils.config import load_config`.

### Section 3: Code
```python
"""Utilities package: configuration loading."""

from src.utils.config import Config, load_config

__all__ = ["Config", "load_config"]
```

### Section 4: Walkthrough
- Pure re-export.

### Section 5: Verification
```bash
python -c "from src.utils.config import load_config; print('ok')"
# Expect: ok
```

---

## `src/utils/config.py`

### Section 1: Header
- **Path:** `src/utils/config.py`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Load YAML into an object with **dot-notation** access (`config.model.decoder.d_model`) and sane defaults, so every script reads hyperparameters from one source of truth (the contract table). Extended from the old `code/` with the new transformer/training keys.

### Section 2: Concept Lesson

**Why a config object, not scattered constants.** The contract table at the top says "if config says `d_model: 384`, every file uses `config.model.decoder.d_model`." Centralizing hyperparameters means changing the model is a YAML edit, not a code hunt — and `from_config` (in `occupancy_network.py`) reads exactly these fields. Dot-notation (`config.training.lr`) reads better than `config['training']['lr']` and matches the old `code/`'s `ConfigDict`. We support nested defaults + deep-merge so a partial YAML (like `dinov2_vitb.yaml`, which overrides only a few keys) inherits everything else.

No notebook covers config systems (they hardcode hyperparameters in cells); this is standard project infrastructure, kept from the old `code/` and extended.

### Section 3: The Complete Code
```python
"""YAML configuration with dot-notation access and deep-merged defaults."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigDict(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc
        return ConfigDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# Full default configuration. A YAML file overrides only the keys it specifies.
DEFAULTS: Dict[str, Any] = {
    "model": {
        "encoder": {
            "variant": "dinov2_vits14",
            "freeze": True,
            "use_lora": False,
            "lora_r": 16,
            "lora_alpha": 16,
        },
        "decoder": {
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 4,
            "num_bands": 10,
            "dropout": 0.1,
            "ffn_mult": 4,
        },
    },
    "training": {
        "batch_size": 16,
        "num_points": 2048,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "num_epochs": 100,
        "warmup_steps": 500,
        "grad_clip": 1.0,
        "amp": True,
        "num_workers": 4,
        "augment": True,
    },
    "inference": {
        "grid_resolution": 64,
        "threshold": 0.5,
        "query_batch_size": 100000,
    },
    "data": {
        "image_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Recursively merge ``update`` into ``base`` in place."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class Config:
    """Parsed configuration exposing ``.model``, ``.training``, ``.inference``, ``.data``."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """Start from DEFAULTS and deep-merge an optional override dict."""
        import copy
        self._config = copy.deepcopy(DEFAULTS)
        if config_dict:
            _deep_update(self._config, config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a YAML file (missing file => pure defaults)."""
        config = cls()
        yaml_path = Path(path)
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if loaded:
                _deep_update(config._config, loaded)
        return config

    @property
    def model(self) -> ConfigDict:
        return ConfigDict(self._config["model"])

    @property
    def training(self) -> ConfigDict:
        return ConfigDict(self._config["training"])

    @property
    def inference(self) -> ConfigDict:
        return ConfigDict(self._config["inference"])

    @property
    def data(self) -> ConfigDict:
        return ConfigDict(self._config["data"])

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict copy (for logging / checkpointing)."""
        import copy
        return copy.deepcopy(self._config)


def load_config(path: str = "configs/default.yaml") -> Config:
    """Convenience loader."""
    return Config.from_yaml(path)


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    assert cfg.model.encoder.variant == "dinov2_vits14"
    assert cfg.model.decoder.d_model == 384
    assert cfg.model.decoder.d_model % cfg.model.decoder.n_heads == 0
    print("config.py self-test:", cfg.model.encoder.variant,
          "d_model", cfg.model.decoder.d_model, "heads", cfg.model.decoder.n_heads)
```

### Section 4: Line-by-Line Walkthrough
- **`ConfigDict.__getattr__` wraps nested dicts on access** so `config.model.decoder.d_model` chains naturally — each level returns a fresh `ConfigDict`. It raises `AttributeError` (not `KeyError`) for missing keys, which is the Pythonic contract for attribute access and gives clear errors.
- **`DEFAULTS` is the complete schema**; YAML files override subsets. This is what lets `dinov2_vitb.yaml` specify only 4 keys and inherit the other ~25 — the deep-merge fills the rest.
- **`copy.deepcopy(DEFAULTS)`** prevents one `Config` instance from mutating the shared module-level defaults (a subtle aliasing bug the old `code/`'s `.copy()` could hit on nested dicts).
- **`d_model % n_heads == 0` asserted in the self-test** — this is the single most important config invariant (attention head split); catching it here prevents a confusing crash inside `MultiHeadCrossAttention`.
- **`to_dict`** is used by `train.py` to log the full config to TensorBoard and stamp checkpoints.

### Section 5: Verification
```bash
python -m src.utils.config
# Expect: config.py self-test: dinov2_vits14 d_model 384 heads 6
```

---

## `configs/default.yaml`

### Section 1: Header
- **Path:** `configs/default.yaml`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** The default hyperparameters — DINOv2 ViT-S/14, d_model 384. Mirrors `DEFAULTS` so the YAML is explicit and editable without touching code.

### Section 2: Concept
A readable, version-controlled record of every knob. Students edit this, not the Python. Values match the interface contract exactly.

### Section 3: The Complete Code
```yaml
# 3DScan default configuration — DINOv2 ViT-S/14 + cross-attention decoder.

model:
  encoder:
    variant: dinov2_vits14   # embed_dim 384, patch 14 -> 257 tokens for 224px
    freeze: true             # feature-extraction mode (transfer learning, notebook 05)
    use_lora: false
    lora_r: 16
    lora_alpha: 16
  decoder:
    d_model: 384             # must match (or be projected from) encoder embed_dim
    n_heads: 6               # 384 / 6 = 64 per head
    n_layers: 4
    num_bands: 10            # Fourier coord encoding -> 3 + 3*2*10 = 63 dims
    dropout: 0.1
    ffn_mult: 4

training:
  batch_size: 16
  num_points: 2048
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 100
  warmup_steps: 500
  grad_clip: 1.0
  amp: true
  num_workers: 4
  augment: true

inference:
  grid_resolution: 64
  threshold: 0.5
  query_batch_size: 100000

data:
  image_size: 224            # must be divisible by 14 (ViT patch size)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

### Section 4: Walkthrough
- Every value equals the contract table; comments explain the non-obvious constraints (token count, head split, Fourier dim, %14).

### Section 5: Verification
```bash
python -c "from src.utils.config import load_config; c=load_config('configs/default.yaml'); \
print(c.model.decoder.d_model, c.training.warmup_steps)"
# Expect: 384 500
```

---

## `configs/dinov2_vitb.yaml`

### Section 1: Header
- **Path:** `configs/dinov2_vitb.yaml`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Override config that swaps to the larger **ViT-B/14** backbone (embed_dim 768). Demonstrates that the architecture scales with a 4-key change, thanks to the deep-merge and the decoder's `token_dim→d_model` projection.

### Section 2: Concept
Because `Config` deep-merges over `DEFAULTS`, this file specifies *only what differs*. ViT-B has embed_dim 768, so `d_model: 768` and `n_heads: 12` (768/12 = 64). Everything else (training, inference, data) inherits the defaults.

### Section 3: The Complete Code
```yaml
# ViT-B/14 override: larger, sharper features. Inherits all other defaults.

model:
  encoder:
    variant: dinov2_vitb14   # embed_dim 768
  decoder:
    d_model: 768             # match the bigger token width
    n_heads: 12              # 768 / 12 = 64 per head
```

### Section 4: Walkthrough
- Only the encoder variant and the two decoder dims change; the `token_dim` the decoder receives (768) is read from the encoder's `embed_dim` at construction, so nothing else needs editing.

### Section 5: Verification
```bash
python -c "from src.utils.config import load_config; c=load_config('configs/dinov2_vitb.yaml'); \
print(c.model.encoder.variant, c.model.decoder.d_model, c.model.decoder.n_heads, c.training.batch_size)"
# Expect: dinov2_vitb14 768 12 16    (batch_size inherited from defaults)
```

# Training & Inference

## `train.py`

### Section 1: Header
- **Path:** `train.py` (repo root)
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The training driver. AdamW + linear-warmup→cosine schedule + gradient clipping + mixed precision (AMP) + TensorBoard logging + periodic validation IoU + checkpointing. Optional backbone freeze/LoRA. This is **SYNC 2**.

### Section 2: Concept Lesson

**AdamW over Adam — decoupled weight decay.** The course used `Adam`/`AdamW` interchangeably (`03.2` used `AdamW`). The difference: plain `Adam` with `weight_decay` folds the L2 penalty *into* the gradient, where the adaptive per-parameter scaling distorts it; **AdamW** applies weight decay *separately* (decoupled), as a clean shrink of the weights. For transformers this matters — proper weight decay is a key regularizer, and AdamW is the universal default (BERT, ViT, GPT all use it). We decay the decoder weights but **exclude LayerNorm/bias** from decay (standard practice — decaying norm gains/biases hurts).

**Warmup — what it prevents.** At step 0 the decoder weights are random, so the first gradients are large and noisy. Adam's running variance estimates are also uninitialized. Hitting full learning rate immediately can blow up attention logits (softmax saturates) and destabilize training. **Linear warmup** ramps the LR from 0 → target over `warmup_steps`, letting Adam's statistics settle and the attention maps form gently. After warmup we **cosine-decay** to 0 (the smooth schedule from the old `code/`'s `CosineAnnealingLR`, now with a warmup prefix). This warmup+cosine combo is the standard transformer recipe.

**Gradient clipping with attention.** `06.3` and `07.1` already clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. Attention makes this *more* important: a single saturated softmax can produce a huge gradient spike that, unclipped, throws the weights into a bad region (loss → NaN). Clipping the global gradient norm to 1.0 caps these spikes. We borrow the notebook's exact call.

**Mixed precision (AMP).** `autocast` runs matmuls/attention in fp16/bf16 (faster, less memory) while keeping a master fp32 copy; `GradScaler` scales the loss to prevent fp16 gradient underflow, then unscales before the optimizer step (and before clipping). This is new versus the notebooks (which trained small models in fp32) but standard for transformers on GPU.

**Freeze / LoRA.** With `freeze: true` the ViT is frozen (transfer learning, notebook 05) and only the decoder trains — fast, stable, the default. `use_lora: true` instead injects trainable low-rank adapters into the ViT attention (notebook 11) for a quality bump at low cost.

**Borrowed / adapted / new.**
- *Borrowed:* the `train_step`/`validate` loop skeleton (03.4, 06.3), `clip_grad_norm_` (06.3/07.1), cosine schedule (old `code/`), checkpoint save/best logic (old `code/`).
- *Adapted:* `Adam` → `AdamW` with no-decay groups; cosine → warmup+cosine.
- *New:* AMP (`autocast`/`GradScaler`), TensorBoard, validation IoU metric, freeze/LoRA flags.

### Section 3: The Complete Code
```python
"""Train the DINOv2 + cross-attention occupancy network.

AdamW + warmup->cosine LR + grad clipping (notebooks 06.3/07.1) + AMP +
TensorBoard. Backbone frozen by default (transfer learning, notebook 05).
"""

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import get_dataloader
from src.model import OccupancyNetwork
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (override a few config values)."""
    p = argparse.ArgumentParser(description="Train 3DScan occupancy network")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """AdamW with weight decay applied to weights but not to bias/LayerNorm."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):  # LayerNorm gains & biases
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(groups, lr=lr)


def build_scheduler(
    optimizer: optim.Optimizer, warmup_steps: int, total_steps: int
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_steps`` then cosine decay to 0 over the rest."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple:
    """Return (mean val loss, mean val IoU) over the loader."""
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        points = batch["points"].to(device)
        occupancy = batch["occupancy"].to(device)
        logits = model(images, points)
        total_loss += criterion(logits, occupancy).item()
        pred = (torch.sigmoid(logits) > 0.5)
        gt = (occupancy > 0.5)
        inter = (pred & gt).sum(dim=1).float()
        union = (pred | gt).sum(dim=1).float().clamp(min=1.0)
        total_iou += (inter / union).mean().item()
        n += 1
    return total_loss / max(n, 1), total_iou / max(n, 1)


def train_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch, grad_clip, use_amp
) -> float:
    """Run one training epoch; returns mean loss."""
    model.train()
    # Keep a frozen backbone in eval mode even during training so it never
    # updates internal stats; frozen params have requires_grad=False regardless.
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.backbone.eval()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        points = batch["points"].to(device)
        occupancy = batch["occupancy"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, points)
            loss = criterion(logits, occupancy)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                   # unscale before clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n, 1)


def main() -> None:
    """Train, validate, checkpoint."""
    args = parse_args()
    config = load_config(args.config)

    epochs = args.epochs or config.training.num_epochs
    batch_size = args.batch_size or config.training.batch_size
    lr = args.lr or config.training.learning_rate
    use_amp = bool(config.training.amp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp} | epochs: {epochs} | bs: {batch_size} | lr: {lr}")

    model = OccupancyNetwork.from_config(config).to(device)
    print(f"Trainable params: {model.get_num_params(trainable_only=True):,} / "
          f"{model.get_num_params(trainable_only=False):,} total")

    train_loader = get_dataloader(
        root=args.data_root, split="train", batch_size=batch_size,
        num_workers=config.training.num_workers, num_points=config.training.num_points,
        augment=config.training.augment, max_samples=args.max_samples,
    )
    val_loader = get_dataloader(
        root=args.data_root, split="val", batch_size=batch_size,
        num_workers=config.training.num_workers, num_points=config.training.num_points,
        augment=False, max_samples=(args.max_samples // 5 if args.max_samples else None),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, lr, config.training.weight_decay)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, config.training.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_iou = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            epoch + 1, config.training.grad_clip, use_amp,
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("iou/val", val_iou, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | val IoU {val_iou:.4f}")

        ckpt = {
            "epoch": epoch, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": model.config_snapshot, "best_iou": best_iou,
            "train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  New best (val IoU {val_iou:.4f}) -> best.pt")

    writer.close()
    print(f"Done. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
```

### Section 4: Line-by-Line Walkthrough
- **`build_optimizer` splits params into decay / no-decay.** Any 1-D parameter (LayerNorm weight, all biases) goes in the `weight_decay=0` group. This is the standard "don't decay norms/biases" rule that meaningfully improves transformer training; plain `optim.Adam(model.parameters())` (old `code/`) couldn't express it.
- **`build_scheduler` returns a `LambdaLR`** computing the multiplier per **optimizer step** (not per epoch) — warmup is measured in steps. `min(progress, 1.0)` clamps the cosine so resuming past `total_steps` doesn't push LR negative.
- **AMP ordering is critical:** `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step` → `scaler.update`. We **must unscale before clipping**, otherwise we'd clip the *scaled* gradients (wrong norm). This is the one AMP subtlety to get right.
- **`optimizer.zero_grad(set_to_none=True)`** frees gradient memory and is slightly faster than zeroing in place.
- **Validation IoU, not just loss.** The old `code/` checkpointed on val *loss*; we checkpoint on val **IoU** because IoU is the metric we actually care about (loss and IoU can diverge). The in-loop IoU is the cheap point-wise version (full geometric eval lives in `evaluate.py`).
- **`scheduler.state_dict()` is saved/restored** so `--resume` continues the warmup/cosine curve exactly rather than restarting it.
- **`GradScaler(enabled=use_amp)` and `use_amp = ... and device.type=='cuda'`** — AMP is a no-op on CPU, so we disable it there; the same code runs on both.
- **The frozen-backbone eval guard.** After `model.train()`, if no encoder param requires grad (i.e. the backbone is frozen), we put `model.encoder.backbone` back in `eval()` mode. `model.train()` flips the *whole* tree to train mode; for a frozen backbone we don't want any of its internal modules in train mode. Frozen params never update regardless (their `requires_grad=False`), so this only affects mode-dependent sub-layers — a correctness belt-and-braces.
- **`BCEWithLogitsLoss` not `BCELoss`+sigmoid:** it fuses sigmoid and BCE via a log-sum-exp formulation that is numerically stable (no `log(0)` when a prediction saturates). This is exactly why the decoder returns logits — covered in `03.3 Classification`.

### Section 5: Verification
```bash
# Smoke train (SYNC 2). Needs data; tiny subset + few epochs.
python train.py --config configs/default.yaml --data-root <DATA> --max-samples 8 --epochs 3
# Expect: 3 epoch lines; train loss trends down; checkpoints/last.pt and best.pt written;
#         a checkpoints/tb/ TensorBoard log directory appears.
tensorboard --logdir checkpoints/tb   # optional: view curves
```

---

## `inference.py`

### Section 1: Header
- **Path:** `inference.py` (repo root)
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The end-to-end demo: one image → STL. Loads a checkpoint, preprocesses the image, generates the occupancy grid, runs Marching Cubes + `make_printable`, and saves the STL. The spec's full pipeline in one script.

### Section 2: Concept Lesson

**Inference = the forward path with no training machinery.** `model.eval()`, `torch.no_grad()` (already inside `generate_occupancy_grid`), no optimizer, no loss. We reuse the *same* `ImagePreprocessor` (no augmentation) so the image is normalized identically to training — a mismatch here is the most common "works in training, garbage at inference" bug. Then the mesh back-end (Dev B's files) turns probabilities into a printable solid. This script is the human-facing artifact: feed a photo, get a printable `.stl`.

### Section 3: The Complete Code
```python
"""Inference: single image -> occupancy grid -> Marching Cubes -> printable STL."""

import argparse

import torch
from PIL import Image

from src.data.preprocessing import ImagePreprocessor
from src.mesh import extract_mesh, save_mesh
from src.model import OccupancyNetwork
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="3DScan: image -> STL")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Input image path")
    p.add_argument("--output", type=str, required=True, help="Output STL path")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    """Run the full image->STL pipeline."""
    args = parse_args()
    config = load_config(args.config)
    resolution = args.resolution or config.inference.grid_resolution
    threshold = args.threshold or config.inference.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | resolution {resolution} | threshold {threshold}")

    model = OccupancyNetwork.from_checkpoint(args.checkpoint, map_location=str(device)).to(device)
    model.eval()

    preprocessor = ImagePreprocessor(image_size=config.data.image_size)
    image = Image.open(args.input).convert("RGB")
    image_tensor = preprocessor(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    print("Generating occupancy grid...")
    grid = model.generate_occupancy_grid(
        image_tensor, resolution=resolution,
        query_batch_size=config.inference.query_batch_size,
    )
    print(f"Grid {grid.shape} | prob range [{grid.min():.3f}, {grid.max():.3f}]")

    print("Extracting + cleaning mesh...")
    mesh = extract_mesh(grid, threshold=threshold, pad=True, postprocess=True)
    if mesh is None:
        print("No surface found. Try a lower --threshold (e.g. 0.3).")
        return

    save_mesh(mesh, args.output)
    print(f"Done -> {args.output} | watertight={mesh.is_watertight}")


if __name__ == "__main__":
    main()
```

### Section 4: Line-by-Line Walkthrough
- **`ImagePreprocessor(image_size=config.data.image_size)` with `augment` defaulting off** guarantees inference normalization matches training's eval transform exactly.
- **`.unsqueeze(0)`** adds the batch dim → `(1,3,224,224)`; `generate_occupancy_grid` requires batch size 1 (it raises otherwise).
- **`query_batch_size` from config** so a low-VRAM machine can shrink the per-chunk point count without code edits.
- **`mesh is None` branch** gives an actionable hint (lower the threshold) instead of a stack trace — the grid may never cross 0.5 for an under-trained model.
- **`pad=True, postprocess=True`** ensures the saved STL is the watertight, printable version (the spec's deliverable), and watertightness is reported.
- **`from_checkpoint`** rebuilds the exact architecture from the checkpoint's config snapshot — inference never needs `--config` to match the trained model's dims (the `--config` here only supplies inference defaults like resolution).

### Section 5: Verification
```bash
python inference.py --checkpoint checkpoints/best.pt --input examples/chair.jpg --output out.stl
# Expect: Grid (64,64,64) | prob range [...]; "Exported STL: out.stl | V=... F=... watertight=True"
# Open out.stl in any mesh viewer (or a slicer) to confirm a recognizable, closed shape.
```

# Tests

## `tests/__init__.py`

### Section 1: Header
- **Path:** `tests/__init__.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** Marks `tests/` as a package so `pytest` and relative imports resolve cleanly.

### Section 2: Concept
Empty package marker, same rationale as `src/__init__.py`.

### Section 3: Code
```python
"""Test package for 3DScan."""
```

### Section 4: Walkthrough
- Empty by design.

### Section 5: Verification
```bash
python -c "import tests; print('ok')"
# Expect: ok
```

---

## `tests/test_shapes.py`

### Section 1: Header
- **Path:** `tests/test_shapes.py`
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** The interface guard. Asserts every cross-file shape contract from the table at the top: Fourier dim, attention block, decoder I/O (ViT-S **and** ViT-B dims), encoder→decoder token width, grid→mesh, metrics, postprocess. CPU-only and fast; the one DINOv2-dependent test self-skips when offline.

### Section 2: Concept Lesson

**Shape tests are cheap insurance against integration drift.** Two developers building against a contract will, sooner or later, disagree about a dimension. These tests fail the instant `encoder.embed_dim` and `decoder.token_dim` diverge, or a refactor changes the Fourier dim. They use **tiny synthetic tensors** so they run in milliseconds and need no GPU/data — except the single end-to-end test that actually loads DINOv2, which we **skip gracefully** if the weights can't be fetched (so CI stays green offline). This mirrors the assertion-driven self-tests every module already carries, gathered into one `pytest` suite (the testing discipline implied by `pytest` being in the course `requirements.txt`).

### Section 3: The Complete Code
```python
"""Shape-contract tests covering every cross-file interface."""

import numpy as np
import pytest
import torch

from src.model.layers import (
    CrossAttentionBlock,
    FeedForward,
    FourierPositionalEncoding,
    MultiHeadCrossAttention,
)
from src.model.decoder import CrossAttentionOccupancyDecoder
from src.model.encoder import _DINOV2_EMBED_DIM
from src.mesh import extract_mesh, make_printable
from src.eval.metrics import chamfer_l1, f_score, normal_consistency, volumetric_iou
from src.utils.config import load_config


# ---------------- layers.py ----------------

def test_fourier_output_dim():
    pe = FourierPositionalEncoding(in_dim=3, num_bands=10)
    assert pe.output_dim == 63
    out = pe(torch.rand(2, 128, 3) * 2 - 1)
    assert out.shape == (2, 128, 63)


def test_multihead_cross_attention_shape():
    attn = MultiHeadCrossAttention(d_model=384, n_heads=6, dropout=0.0)
    q = torch.randn(2, 100, 384)
    kv = torch.randn(2, 257, 384)
    assert attn(q, kv).shape == (2, 100, 384)


def test_attention_requires_divisible_heads():
    with pytest.raises(ValueError):
        MultiHeadCrossAttention(d_model=384, n_heads=7)


def test_feedforward_shape():
    ff = FeedForward(d_model=384, mult=4, dropout=0.0)
    assert ff(torch.randn(2, 50, 384)).shape == (2, 50, 384)


def test_cross_attention_block_shape():
    block = CrossAttentionBlock(d_model=384, n_heads=6, dropout=0.0)
    out = block(torch.randn(2, 100, 384), torch.randn(2, 257, 384))
    assert out.shape == (2, 100, 384)


# ---------------- decoder.py ----------------

def test_decoder_vits_shapes():
    """Default ViT-S dims: token_dim 384 -> logits (B, N, 1)."""
    dec = CrossAttentionOccupancyDecoder(token_dim=384, d_model=384, n_heads=6, n_layers=4)
    pts = torch.rand(2, 2048, 3) * 2 - 1
    tokens = torch.randn(2, 257, 384)
    assert dec(pts, tokens).shape == (2, 2048, 1)


def test_decoder_vitb_shapes():
    """ViT-B override: token_dim 768, d_model 768, 12 heads still yields (B, N, 1)."""
    dec = CrossAttentionOccupancyDecoder(token_dim=768, d_model=768, n_heads=12, n_layers=4)
    pts = torch.rand(2, 512, 3) * 2 - 1
    tokens = torch.randn(2, 257, 768)
    assert dec(pts, tokens).shape == (2, 512, 1)


def test_decoder_arbitrary_point_count():
    """Point-wise independence: any N works (Perceiver-style)."""
    dec = CrossAttentionOccupancyDecoder(token_dim=384, d_model=384, n_heads=6, n_layers=2)
    for n in (1, 37, 4096):
        out = dec(torch.rand(1, n, 3) * 2 - 1, torch.randn(1, 257, 384))
        assert out.shape == (1, n, 1)


# ---------------- config <-> encoder/decoder consistency ----------------

def test_config_matches_encoder_embed_dim():
    """The contract: decoder token_dim must equal the encoder's embed_dim."""
    cfg = load_config("configs/default.yaml")
    variant = cfg.model.encoder.variant
    assert cfg.model.decoder.d_model == _DINOV2_EMBED_DIM[variant]
    assert cfg.model.decoder.d_model % cfg.model.decoder.n_heads == 0


def test_config_vitb_consistency():
    cfg = load_config("configs/dinov2_vitb.yaml")
    assert cfg.model.decoder.d_model == _DINOV2_EMBED_DIM[cfg.model.encoder.variant] == 768
    assert cfg.model.decoder.d_model % cfg.model.decoder.n_heads == 0


# ---------------- grid -> mesh interface ----------------

def test_grid_to_mesh_watertight():
    r = 32
    ax = np.linspace(-1, 1, r)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sphere = (np.sqrt(gx**2 + gy**2 + gz**2) < 0.8).astype(np.float32)
    mesh = extract_mesh(sphere, threshold=0.5)
    assert mesh is not None
    assert len(mesh.faces) > 0
    assert mesh.is_watertight


def test_empty_grid_returns_none():
    grid = np.zeros((16, 16, 16), dtype=np.float32)  # never crosses 0.5
    assert extract_mesh(grid, threshold=0.5) is None


# ---------------- metrics ----------------

def test_metrics_perfect_match():
    rng = np.random.RandomState(0)
    pts = rng.rand(300, 3).astype(np.float32)
    normals = rng.randn(300, 3).astype(np.float32)
    assert chamfer_l1(pts, pts) == 0.0
    assert abs(normal_consistency(pts, normals, pts, normals) - 1.0) < 1e-5
    assert f_score(pts, pts, tau=0.01) == 1.0


def test_iou_known_value():
    pred = np.array([0.9, 0.9, 0.1, 0.1])
    gt = np.array([1.0, 0.0, 0.0, 1.0])
    assert abs(volumetric_iou(pred, gt) - (1 / 3)) < 1e-6


# ---------------- end-to-end (needs DINOv2; skips offline) ----------------

def test_full_model_forward_if_available():
    """Build OccupancyNetwork from config and run a forward pass.

    Skips if DINOv2 weights cannot be downloaded (offline CI).
    """
    try:
        from src.model import OccupancyNetwork
        cfg = load_config("configs/default.yaml")
        model = OccupancyNetwork.from_config(cfg).eval()
    except Exception as exc:  # noqa: BLE001 - network/hub unavailable
        pytest.skip(f"DINOv2 unavailable: {exc}")

    img = torch.randn(2, 3, 224, 224)
    pts = torch.rand(2, 256, 3) * 2 - 1
    with torch.no_grad():
        tokens = model.encode(img)
        logits = model(img, pts)
    assert tokens.shape == (2, 257, cfg.model.decoder.d_model)
    assert logits.shape == (2, 256, 1)

    grid = model.generate_occupancy_grid(img[:1], resolution=16, query_batch_size=512)
    assert grid.shape == (16, 16, 16)
    assert 0.0 <= float(grid.min()) and float(grid.max()) <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

### Section 4: Line-by-Line Walkthrough
- **`from src.model.encoder import _DINOV2_EMBED_DIM`** lets us assert `config.d_model == embed_dim[variant]` **without downloading** the backbone — the most important contract (token width) is checked offline.
- **Separate ViT-S and ViT-B decoder tests** prove the "4-key config swap" claim: both `(384, 6)` and `(768, 12)` produce valid `(B, N, 1)` outputs.
- **`test_decoder_arbitrary_point_count`** encodes the Perceiver-style guarantee in a test: N ∈ {1, 37, 4096} all work, proving there is no hidden self-attention coupling points.
- **`test_attention_requires_divisible_heads`** locks the `d_model % n_heads` invariant with an expected `ValueError`.
- **`test_grid_to_mesh_watertight` + `test_empty_grid_returns_none`** cover both branches of the grid→mesh boundary (surface present vs. absent).
- **The end-to-end test is the only one touching the network**, and it `pytest.skip`s on any load failure, so the suite is green offline and thorough online. It also re-checks the token width equals `config.d_model` — the encoder↔decoder seam — at runtime.
- **`if __name__ == '__main__': pytest.main([__file__, '-v'])`** lets you run the file directly or via `pytest`.

### Section 5: Verification
```bash
pytest tests/ -v
# Expect: all tests PASS; the end-to-end test PASSES (if DINOv2 downloads) or SKIPS (offline).
```
> This is **SYNC 3**: the full interface suite green is the gate before declaring the rebuild done.

# Infrastructure

## `requirements.txt`

### Section 1: Header
- **Path:** `requirements.txt`
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** Pinned dependency manifest. Adds the transformer + evaluation dependencies on top of the old `code/`'s mesh/data stack.

### Section 2: Concept Lesson
**Why these pins.** `torch>=2.2` gives us `F.scaled_dot_product_attention` with FlashAttention dispatch (used in `layers.py`). DINOv2 loads via `torch.hub` — **no extra package** required (the `transformers`/`timm`/`peft`/`einops` lines are optional: `transformers` only if you prefer the HuggingFace loader, `peft` only if you enable LoRA). `numpy<2.0` is pinned because `trimesh`/`scikit-image` wheels in this range expect the NumPy 1.x ABI. `scipy` provides `cKDTree` for the metrics. `tensorboard` backs `train.py` logging. This matches the course stack (torch 2.x, numpy, scipy, scikit-image) and `REBUILD_PLAN.md` §7.

### Section 3: The Complete Code
```text
# --- core deep learning ---
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24,<2.0          # 1.x ABI for trimesh / scikit-image
scipy>=1.10.0             # cKDTree for Chamfer / F-score / Normal Consistency

# --- DINOv2 encoder (loaded via torch.hub; no extra package needed) ---
# Optional alternatives / extras:
einops>=0.7.0             # optional: cleaner tensor reshapes
# transformers>=4.40.0    # optional: HuggingFace DINOv2 loader instead of torch.hub
# timm>=0.9.16            # optional: extra ViT utilities
# peft>=0.10.0            # optional: LoRA fine-tuning of the backbone

# --- mesh / 3D ---
scikit-image>=0.21.0      # marching_cubes
trimesh>=4.0.0
rtree>=1.0.0              # trimesh.contains / proximity backend

# --- data / image ---
Pillow>=9.5.0
PyYAML>=6.0

# --- training / logging ---
tqdm>=4.65.0
tensorboard>=2.15.0
matplotlib>=3.7.0

# --- tests ---
pytest>=7.3.0
```

### Section 4: Walkthrough
- Required vs optional is explicit: the default path needs only the uncommented lines. `einops` is uncommented as a convenience but the provided code does not strictly require it (all reshapes use `.view`/`.transpose`); leave it in for future modules.
- `peft` is commented because the default config has `use_lora: false`; uncomment only if you flip that.

### Section 5: Verification
```bash
pip install -r requirements.txt
python -c "import torch, torchvision, numpy, scipy, skimage, trimesh, yaml, tqdm; print('deps ok')"
# Expect: deps ok
```

---

## `verify_setup.py`

### Section 1: Header
- **Path:** `verify_setup.py` (repo root)
- **Developer:** 👤 **DEVELOPER A**
- **Purpose:** One-shot environment check: PyTorch + CUDA status, and that DINOv2 loads and emits the expected `(B, 257, 384)` tokens. Run at **SYNC 1** and whenever the environment changes.

### Section 2: Concept Lesson
**Fail fast on environment, not 20 minutes into training.** The single biggest external risk is DINOv2 fetching from `torch.hub`. This script isolates that risk: it downloads the backbone once, verifies token shapes against the contract, and reports CUDA. If this passes, every model file can rely on the encoder. It's the project's smoke detector.

### Section 3: The Complete Code
```python
"""Verify the environment: PyTorch, CUDA, and DINOv2 token shapes."""

import sys

import torch


def main() -> int:
    """Run checks; return process exit code (0 = all good)."""
    print(f"PyTorch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  torch CUDA: {torch.version.cuda}")

    try:
        from src.model.encoder import DINOv2Encoder
        enc = DINOv2Encoder("dinov2_vits14", freeze=True).eval()
        with torch.no_grad():
            tokens = enc(torch.randn(2, 3, 224, 224))
        print(f"DINOv2 tokens: {tuple(tokens.shape)} (embed_dim {enc.embed_dim})")
        assert tokens.shape == (2, 257, 384), f"unexpected token shape {tokens.shape}"
    except Exception as exc:  # noqa: BLE001
        print(f"DINOv2 check FAILED: {exc}")
        return 1

    print("Setup OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Section 4: Line-by-Line Walkthrough
- **Returns an exit code** (`sys.exit(main())`) so CI can gate on it (`0` pass / `1` fail).
- **Asserts the exact `(2, 257, 384)` contract** — if a future DINOv2 release changes token layout, this catches it before model code does.
- **Wrapped in try/except** so a network failure prints a clean diagnosis instead of a traceback.
- **Prints CUDA device + toolkit version** to confirm the GPU path is live (AMP in `train.py` depends on it).

### Section 5: Verification
```bash
python verify_setup.py
# Expect (online): PyTorch: 2.x ; CUDA available: True/False ; DINOv2 tokens: (2, 257, 384) (embed_dim 384) ; Setup OK.
# Exit code 0.
```

---

## `README.md`

### Section 1: Header
- **Path:** `README.md` (repo root)
- **Developer:** 👤 **DEVELOPER B**
- **Purpose:** The front door: architecture diagram, install, train, evaluate, infer commands, and a layout map. Replaces the old two-line stub.

### Section 2: Concept Lesson
A README is the project's API for humans. It should let a newcomer go from clone → trained model → STL without reading source. We mirror the command set the verification sections use.

### Section 3: The Complete Code
````markdown
# 3DScan — Single-Image 3D Reconstruction (DINOv2 + Cross-Attention)

Turn one 2D photo into a 3D-printable STL mesh. A pretrained **DINOv2 Vision
Transformer** encodes the image into patch tokens; a **cross-attention decoder**
queries 3D coordinates against those tokens to predict occupancy; **Marching
Cubes** + **Trimesh** turn the occupancy field into a watertight STL.

## Architecture

```
image (B,3,224,224)
  → DINOv2Encoder (frozen ViT-S/14) → tokens (B, 257, 384)
points (B,N,3) → FourierPE(63) → Linear(384) → Q
  → CrossAttentionOccupancyDecoder: 4× [cross-attn(Q=points, K/V=tokens) + FFN]
  → logits (B,N,1)
infer: 64³ grid → sigmoid → Marching Cubes → make_printable → STL
```

See `REBUILD_PLAN.md` (architecture rationale) and `CODE_PLAN.md` (full build guide).

## Install

```bash
pip install -r requirements.txt
python verify_setup.py        # downloads DINOv2 once; checks CUDA + token shapes
```

## Data

ShapeNet in the OccNet/Choy-2016 layout:

```
DATA_ROOT/<category_id>/<model_id>/
    points.npz            # points (P,3) in [-0.5,0.5] + packed occupancies
    pointcloud.npz        # surface points + normals (for evaluation)
    img_choy2016/*.jpg    # rendered views (+ cameras.npz)
```

13 categories: airplane, bench, cabinet, car, chair, display, lamp, speaker,
rifle, sofa, table, telephone, vessel.

## Train

```bash
python train.py --config configs/default.yaml --data-root DATA_ROOT
# bigger backbone:
python train.py --config configs/dinov2_vitb.yaml --data-root DATA_ROOT
# quick smoke test:
python train.py --config configs/default.yaml --data-root DATA_ROOT --max-samples 8 --epochs 3
tensorboard --logdir checkpoints/tb
```

## Evaluate

```bash
python src/eval/evaluate.py --checkpoint checkpoints/best.pt --data-root DATA_ROOT
# -> per-category + mean IoU / Chamfer-L1 / Normal Consistency / F-score
```

## Inference (image → STL)

```bash
python inference.py --checkpoint checkpoints/best.pt --input photo.jpg --output model.stl
```

## Test

```bash
pytest tests/ -v
```

## Layout

```
src/model/    layers, encoder (DINOv2), decoder (cross-attention), occupancy_network
src/data/     ShapeNet dataset + preprocessing
src/mesh/     marching cubes, printable post-process, export
src/eval/     metrics + evaluation driver
src/utils/    config
configs/      default.yaml (ViT-S), dinov2_vitb.yaml (ViT-B)
train.py inference.py verify_setup.py
tests/
```
````

### Section 4: Walkthrough
- Every command shown is one that appears in a file's Section 5 verification — the README is the union of the verified entry points, so it cannot drift from reality.
- The data-layout block documents the `[-0.5, 0.5]` storage convention that `dataset.py` scales to `[-1, 1]` — the one fact a data-wrangler must know.

### Section 5: Verification
```bash
# Render check: open README.md in any Markdown viewer; confirm the code fences and diagram render.
# All commands listed are individually verified in their respective files' Section 5.
```

---

# Appendix: Build Order & Final Checklist

**Dependency-respecting build order (single-developer fallback):**
1. `src/__init__.py`, `src/utils/config.py`, `configs/*.yaml` → config compiles.
2. `src/model/layers.py` → `python -m src.model.layers` passes.
3. `src/model/encoder.py` → `verify_setup.py` passes (downloads DINOv2).
4. `src/model/decoder.py` → `python -m src.model.decoder` passes.
5. `src/model/occupancy_network.py` + `src/model/__init__.py` → SYNC 1 command passes.
6. `src/data/preprocessing.py`, `src/data/dataset.py`, `src/data/__init__.py`.
7. `src/mesh/postprocess.py`, `marching_cubes.py`, `export.py`, `__init__.py`.
8. `train.py` → SYNC 2 smoke train.
9. `src/eval/metrics.py`, `evaluate.py`, `__init__.py`.
10. `inference.py`.
11. `tests/__init__.py`, `tests/test_shapes.py` → SYNC 3 `pytest` green.
12. `requirements.txt`, `verify_setup.py`, `README.md`.

**Final acceptance checklist:**
- [ ] `pip install -r requirements.txt` clean.
- [ ] `python verify_setup.py` → `Setup OK.` (exit 0).
- [ ] `pytest tests/ -v` → all pass (end-to-end test passes or skips offline).
- [ ] `python train.py --data-root DATA --max-samples 8 --epochs 3` → loss decreases, `best.pt` written.
- [ ] `python src/eval/evaluate.py --checkpoint checkpoints/best.pt --data-root DATA --max-samples 16` → metric table.
- [ ] `python inference.py --checkpoint checkpoints/best.pt --input IMG --output out.stl` → watertight STL opens in a viewer.

*End of CODE_PLAN.md.*
