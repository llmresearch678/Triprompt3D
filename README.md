# TRIPROMPT-3D: Deformation-Aware Multimodal Prompting for Robust 3D Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge)](https://pytorch.org)

**A deformation-aware, multimodal prompting framework for robust 3D organ segmentation and probabilistic anatomical reconstruction**

## 🔍 Overview

Robust 3D medical image segmentation remains challenging due to substantial anatomical variability arising from respiration, pathology, and inter-subject morphological differences. Existing prompt-based segmentation approaches primarily model *what* organs look like and *where* they are — but largely ignore **how organs deform**.

**TRIPROMPT-3D** introduces a fully automatic framework that integrates three complementary prompt modalities:

| Prompt | Symbol | Encodes |
|---|---|---|
| **Structural Prompt** | `Qa` | Localized anatomical appearance & spatial geometry from 3D sub-volumes |
| **Text Prompt** | `Qt` | Medical semantic knowledge via ClinicalBERT |
| **Population-Level Deformation Prompt** | `Qd / PDP` | Statistical organ deformation patterns via probabilistic latent representation |

These prompts are fused through a **query-centric TriPrompt Aligner**, enabling cross-modal interaction between appearance, semantics, and physiological deformation during segmentation inference.

### Why Deformation Matters

> Classical methods model shape as static. Real anatomy bends, stretches, and shifts across patients, disease stages, and respiratory phases. TRIPROMPT-3D explicitly captures this variability.

The **Population-Level Deformation Prompt (PDP)** is:
- **Nonlinear** — learned from data, not a fixed linear PCA basis
- **Probabilistic** — models uncertainty via a VAE-style latent space
- **Uncertainty-aware** — adaptively blends image evidence with population priors
- **Architecturally integrated** — directly participates in segmentation query refinement

> *Technically: Linear PCA priors ⊂ Nonlinear PDP latent deformation model*

---

## ✨ Key Contributions

1. **Three-Prompt Architecture** — The first framework to jointly integrate structural, textual, and population-level deformation prompts for 3D medical segmentation.

2. **Probabilistic Deformation Prompt (PDP)** — A conditional VAE that models class-conditional organ deformation as a structured latent variable, enabling uncertainty-adaptive inference.

3. **Reliability Score (αc)** — A confidence measure derived from posterior covariance that gracefully transitions inference from image-conditioned estimates to population priors when image evidence is weak.

4. **Query-Centric TriPrompt Aligner** — A multi-scale cross-modal attention mechanism combining hard spatial routing (Gumbel-Softmax) for structural/deformation prompts with soft attention for semantic prompts.

5. **Theoretical Guarantees** — Formal proofs of (i) local identifiability of deformation representations, (ii) non-redundancy via mutual information analysis, and (iii) prompt interference control via sparse routing bounds.

6. **State-of-the-Art Results** — Consistent improvements on 11 public CT benchmarks, with up to **+7.3% DSC on tumors** and **+5.4% cross-dataset mean DSC** over the best baseline.

---

## 🧠 Method

### Architecture Overview

```
Input 3D CT Volume (H×W×D)
        │
        ▼
┌───────────────────┐
│  Swin-UNETR       │  Shared 3D Backbone
│  Backbone         │  Feature dim C = 256
└────────┬──────────┘
         │  Multi-scale features {fℓ}
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌───────┐  ┌──────────────────────────┐
│  Qa   │  │  Qt   │  │         Qd (PDP)          │
│Struct.│  │ Text  │  │  Population Deformation   │
│Prompt │  │Prompt │  │         Prompt            │
│3D     │  │BERT / │  │  EDEF(cross-subject masks)│
│ResNet │  │CLIP   │  │  + Probabilistic head Gθ  │
└───┬───┘  └───┬───┘  └────────────┬─────────────┘
    │           │                   │
    └───────────┴───────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   TriPrompt Aligner   │
            │  (Cross-modal attn)   │
            │  Hard routing: Qa, Qd │
            │  Soft attn:  Qs, Qt   │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  PromptContextAligner │
            │  Os = Qs + SoftAttn   │
            │     + MaskedAttn(Qa)  │
            │     + MaskedAttn(Qd)  │
            └───────────┬───────────┘
                        │
                        ▼
            Multi-label voxel-wise segmentation
            M ∈ [0,1]^{K×H×W×D}
```

### Uncertainty-Adaptive Deformation Prompt

The PDP operates via a **dual-pathway mechanism**:

**Pathway 1 — Population Prior:**
```
Cross-subject shape masks → EDEF(·) → Q_d,POP  (population deformation token)
```

**Pathway 2 — Image-Conditioned Posterior:**
```
2D Input + Multi-scale features → Gθ → (μ_d, Σ_d)  (mean + covariance)
```

**Adaptive Blending:**
```
αc = exp(−τ · tr(Σ_d))        # reliability score: near 1 = high confidence
Q̃_d = αc · μ_d + (1−αc) · Q_d,POP
```

When image evidence is strong → relies on μ_d.  
When image evidence is weak → defers to population prior Q_d,POP.

### Training Objective

```
L = L_SEG + L_CE + λ1·L_ALIGN(1) + λ2·L_ALIGN(2)
```

Where:
- `L_SEG` — Multi-label Dice loss for volumetric segmentation
- `L_CE` — Cross-entropy loss over class logits  
- `L_ALIGN(1)` — Contrastive segmentation–prompt alignment
- `L_ALIGN(2)` — Contrastive anatomical–textual prompt alignment
- Weights `λ1`, `λ2` are updated via **gradient-norm balancing** every epoch

---

## 📊 Results

### Per-Organ Segmentation (11 CT Datasets)

| Method | Liver DSC | Pancreas DSC | Gallbladder DSC | Tumor DSC |
|---|---|---|---|---|
| nnU-Net | 94.8 | 78.2 | 72.1 | 61.4 |
| Swin-UNETR | 95.3 | 79.8 | 73.9 | 63.2 |
| MedSAM | 95.6 | 80.4 | 74.8 | 64.4 |
| CLIP-Driven | 95.9 | 81.3 | 75.6 | 65.7 |
| SAM-Med3D | 96.1 | 82.1 | 76.4 | 66.8 |
| SegVol | 96.4 | 83.7 | 78.2 | 68.3 |
| **TRIPROMPT-3D (Ours)** | **97.6** | **88.4** | **84.7** | **75.6** |
| **Gain vs. 2nd best** | **+1.2** | **+4.7** | **+6.5** | **+7.3** |

> Performance gains are largest for **deformable, low-contrast structures** where population-level deformation priors provide the greatest benefit.

### Cross-Dataset Generalization (Zero-Shot, No Domain Adaptation)

| Method | FLARE22 Multi | FLARE22 Tumor | MSD Pancreas | MSD Mean |
|---|---|---|---|---|
| SegVol (2nd best) | 88.3 | 60.1 | 74.2 | 84.5 |
| **TRIPROMPT-3D** | **91.4** | **67.8** | **81.3** | **88.8** |
| **Gain** | **+3.1** | **+7.7** | **+7.1** | **+4.3** |

**Overall cross-dataset mean improvement: +5.4% DSC over SegVol**

### Clinical Colorectal Tumor Validation (80 Scans, Stages I–IV)

| Method | Stage I | Stage II | Stage III | Stage IV | Mean |
|---|---|---|---|---|---|
| SegVol | 70.1 | 66.8 | 61.4 | 53.9 | 63.1 |
| **TRIPROMPT-3D** | **78.7** | **75.3** | **70.8** | **63.4** | **72.1** |
| **Gain** | **+8.6** | **+8.5** | **+9.4** | **+9.5** | **+9.0** |

> Gains *widen* with tumor stage — exactly where deformation-induced variability is most severe.

### Uncertainty Calibration (FLARE22)

| Method | ECE ↓ | MCE ↓ | AURC ↓ | Brier Score ↓ |
|---|---|---|---|---|
| nnU-Net | 0.092 | 0.181 | 0.117 | 0.084 |
| SAM-Med3D | 0.061 | 0.124 | 0.086 | 0.054 |
| SegVol | 0.053 | 0.109 | 0.079 | 0.047 |
| **TRIPROMPT-3D** | **0.043** | **0.091** | **0.072** | **0.038** |

### Computational Efficiency (Single A100 GPU)

| Method | Params (M) | FLOPs (G) | VRAM (GB) | Latency (s) |
|---|---|---|---|---|
| nnU-Net | 31.2 | 104.8 | 5.4 | 1.19 |
| SAM-Med3D | 87.2 | 124.3 | 6.1 | 0.94 |
| SegVol | 92.4 | 138.7 | 7.4 | 1.12 |
| **TRIPROMPT-3D** | **54.1** | **131.8** | **6.9** | **1.32** |

> The full PDP module adds only **2.1M parameters** and **0.38s latency** — a < 3% overhead relative to the backbone.

---

## ⚙️ Installation

### 1. Create Environment

```bash
conda create -n triprompt python=3.9 -y
conda activate triprompt
```

### 2. Clone Repository

```bash
git clone https://github.com/llmresearch678/Triprompt.git
cd Triprompt
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Libraries

| Library | Purpose |
|---|---|
| PyTorch ≥ 2.0 | Core deep learning framework |
| MONAI | Medical image transforms & utilities |
| HuggingFace Transformers | ClinicalBERT text encoder |
| NumPy | Numerical operations |
| NiBabel | NIfTI file I/O |
| SimpleITK | Medical image resampling |

---

## 📂 Dataset Preparation

### Supported Datasets

| # | Dataset | Year | Scans | Organs | Notes |
|---|---|---|---|---|---|
| 1 | BTCV | 2015 | 30 | 13 | Multi-organ benchmark |
| 2 | CT-ORG | 2020 | 140 | 6 | Multi-center CT |
| 3 | AbdomenCT-1K | 2021 | 1,000 | 4 | Large abdominal CT |
| 4 | MALB | 2018 | 150 | 4 | Multi-atlas labels |
| 5 | AMOS | 2022 | 500 | 15 | CT/MRI challenge |
| 6 | WORD | 2022 | 150 | 16 | Consistent annotation |
| 7 | Pancreas-CT | 2016 | 82 | 1 | Pancreas benchmark |
| 8 | LiTS | 2023 | 201 | 2 | Liver + tumor |
| 9 | KiTS19 | 2019 | 300 | 2 | Kidney + tumor |
| 10 | MSD | 2022 | ~2,600 | 10 | Multi-task decathlon |
| 11 | Synapse | 2019 | 30 | 8 | Abdominal benchmark |

**Total: > 5,100 scans, 16 organ/tumor categories**

### Expected Directory Structure

```
data/
├── train/
│   ├── images/
│   │     ├── case_0001.nii.gz
│   │     ├── case_0002.nii.gz
│   │     └── ...
│   └── masks/
│         ├── case_0001.nii.gz   # Must match image filenames
│         ├── case_0002.nii.gz
│         └── ...
│
└── test/
    ├── images/
    └── masks/
```

> ✅ Images and masks must have **matching filenames**  
> ✅ Masks can be **binary** or **multi-label**  
> ✅ All volumes are automatically resampled to **1.5×1.5×1.5 mm³** isotropic resolution

### Preprocessing (Applied Automatically)

- Resample to 1.5×1.5×1.5 mm³ via B-spline interpolation
- Hounsfield Unit clipping to `[-1000, 1000]`, normalized to `[0, 1]`
- Region-aware foreground bounding box cropping (20% boundary expansion)
- Training augmentation: random flipping, rotation (±15°), elastic deformation, Gaussian noise, intensity scaling

---

## 🚀 Quick Start

### Training from Scratch

```bash
python train.py \
    --data_dir ./data \
    --output_dir ./checkpoints \
    --epochs 150 \
    --batch_size 2 \
    --lr 1e-4
```

Training behavior:
- Sets deterministic seeds across Python, NumPy, and PyTorch
- Jointly optimizes Dice loss + contrastive alignment losses
- Saves checkpoints every 10 epochs to `checkpoints/`
- Gradient-norm balanced loss weights updated every epoch

### Resume Training from Checkpoint

```python
from utils import load_checkpoint

load_checkpoint(
    checkpoint_path="checkpoints/epoch_50.pth",
    model=model,
    optimizer=optimizer,
    device=device
)
# Training resumes exactly from the saved epoch, including optimizer state
```

### Inference

```bash
python inference.py \
    --checkpoint checkpoints/epoch_150.pth \
    --input_dir ./data/test/images \
    --output_dir ./predictions
```

Inference output:
- Loads trained model checkpoint
- Runs voxel-wise, multi-label segmentation inference
- Saves predictions as **NIfTI (.nii.gz)** files
- Multi-label output channels preserved for fair comparison

---

## 📈 Evaluation

Predictions are compatible with standard medical image segmentation metrics:

| Metric | Description |
|---|---|
| **DSC (%)** | Dice Similarity Coefficient — volumetric overlap |
| **HD95 (mm)** | 95th-percentile Hausdorff Distance — boundary error |
| **ASSD (mm)** | Average Symmetric Surface Distance — boundary accuracy |
| **ECE** | Expected Calibration Error — confidence calibration |

All results report **mean ± std** over test volumes. Statistical significance assessed via **Wilcoxon signed-rank test** (p < 0.01).

---

## 🔬 Ablation Study

All variants trained identically on the 11-dataset corpus:

| Variant | Qa | Qt | Qd | Hard Routing | Alignment | DSC ↑ | HD95 ↓ |
|---|---|---|---|---|---|---|---|
| Structural only | ✓ | — | — | ✓ | ✓ | 87.3±1.4 | 11.4 |
| Text only | — | ✓ | — | ✓ | ✓ | 86.9±1.6 | 12.1 |
| Deformation only | — | — | ✓ | ✓ | ✓ | 87.5±1.5 | 11.8 |
| Qa + Qt | ✓ | ✓ | — | ✓ | ✓ | 88.6±1.2 | 10.8 |
| Qa + Qd | ✓ | — | ✓ | ✓ | ✓ | 89.1±1.1 | 9.8 |
| Qt + Qd | — | ✓ | ✓ | ✓ | ✓ | 88.3±1.3 | 10.6 |
| All (soft routing) | ✓ | ✓ | ✓ | — | ✓ | 89.6±1.0 | 9.4 |
| All (no align. losses) | ✓ | ✓ | ✓ | ✓ | — | 89.2±1.1 | 9.7 |
| **Full TRIPROMPT-3D** | ✓ | ✓ | ✓ | ✓ | ✓ | **90.3±0.8** | **7.8** |

### PDP Module Ablation

| PDP Variant | DSC (%) | HD95 (mm) | ECE |
|---|---|---|---|
| No PDP (dual-prompt baseline) | 88.2 | 10.8 | 0.074 |
| Deterministic PDP | 89.1 | 9.6 | 0.062 |
| Probabilistic PDP (no KL) | 89.6 | 9.1 | 0.053 |
| **Probabilistic PDP (full, ours)** | **90.3** | **7.8** | **0.043** |

---

## 📁 Repository Structure

```
Triprompt/
├── models/
│   ├── backbone.py                # Swin-UNETR backbone (C=256)
│   ├── structural_prompt.py       # Structural Prompt encoder (Qa) — 3D ResNet-18
│   ├── text_prompt.py             # Text Prompt encoder (Qt) — ClinicalBERT
│   ├── deformation_prompt.py      # Deformation Prompt (PDP/Qd) — probabilistic VAE
│   ├── triprompt_aligner.py       # Query-centric TriPrompt Aligner
│   └── triprompt_model.py         # Full TRIPROMPT-3D model
│
├── datasets/
│   └── ct_dataset.py              # Unified 3D CT dataset loader (NIfTI)
│
├── losses/
│   ├── dice_loss.py               # Multi-label Dice loss
│   └── contrastive_alignment.py   # Prompt-query & prompt-prompt alignment
│
├── train.py                       # Training script
├── inference.py                   # Inference & prediction saving
├── utils.py                       # Reproducibility, checkpointing, logging
├── requirements.txt
└── README.md
```

---

## 🔄 Reproducibility

All experiments are fully reproducible:

- ✅ Fixed random seeds across Python, NumPy, and PyTorch
- ✅ Deterministic CuDNN behavior enabled
- ✅ No prompt generation inside dataset loaders
- ✅ Deformation prompts sampled from **different subjects** to prevent data leakage
- ✅ All baselines re-trained on the same 11-dataset corpus with published hyperparameters

### Implementation Details

| Component | Specification |
|---|---|
| Backbone | Swin-UNETR, pre-trained on unlabeled CT, C=256 |
| Text Encoder | BERT-Base fine-tuned on MIMIC-III clinical notes |
| Structural Encoder | Lightweight 3D ResNet-18 variant |
| Deformation Encoder | 4× Conv3D + GAP + MLP → B=128 latent codes |
| Posterior Head Gθ | 4-layer MLP (hidden 512, GELU, LayerNorm), Cholesky covariance |
| Optimizer | AdamW (η=1e-4, cosine decay) |
| Batch size | 2 per GPU |
| Hardware | 4× NVIDIA A100 (80GB each) |
| Training time | ~72 hours / 150 epochs |
| Temperature τ | Annealed 1.0 → 0.07 over 150 epochs |

---

## 📖 Theoretical Highlights

**Theorem 1 (Local Identifiability):** Under mild assumptions on the deformation map (gauge fixing, injectivity, local Lipschitz stability), the latent deformation prompt is uniquely identifiable in the noiseless setting and stable under bounded observation noise, with estimation error bounded by 2γ/m.

**Lemma 1 (Mutual Information Gain):** The conditional MI I(Qs; Qd | Qa, Qt) > 0 is strictly positive for all organs and nearly doubles under severe deformation — proving that PDP contributes *non-redundant* geometric information beyond appearance and text.

**Lemma 2 (Prompt Interference Control):** Hard spatial routing via Gumbel-Softmax bounds prompt interference: ‖MASKEDATTN(Qs, Qa, S)‖_F ≤ Ω√N‖S‖₀, ensuring structural precision scales with routing sparsity.

---

## ⚠️ Limitations

- PDP currently requires **fully annotated 3D masks** during training; semi-supervised extension is future work.
- Evaluation is **CT-centric**; MRI and ultrasound involve different contrast characteristics.
- **Temporal (4D) deformation** for respiratory motion is not yet addressed.
- Gumbel-Softmax introduces gradient variance; control-variate strategies could stabilize training.

---

## 🙏 Acknowledgements

We thank the organizers of BTCV, FLARE22, MSD, LiTS, KiTS19, AMOS, WORD, AbdomenCT-1K, CT-ORG, and Pancreas-CT for making their datasets publicly available. We also acknowledge the open-source contributions of nnU-Net, Swin-UNETR, SAM-Med3D, SegVol, and CLIP-Driven, which formed the comparison baselines for this work.

---

<div align="center">

**TRIPROMPT-3D** | Deformation-Aware · Uncertainty-Calibrated · Clinically Robust

*Bridging appearance, semantics, and physiological variability for trustworthy 3D medical image segmentation*

</div>
