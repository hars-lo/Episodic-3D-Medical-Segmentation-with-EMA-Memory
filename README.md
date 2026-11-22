# Episodic 3D Medical Segmentation with EMA Memory

An independent, IRIS-inspired framework for episodic 3D medical image segmentation, combining support-conditioned task representations with dataset-specific Exponential Moving Average (EMA) memory for task retrieval.

Evaluated on ACDC cardiac MRI and MSD Brain Tumour MRI, the framework investigates how reference-guided segmentation and persistent task memory can support segmentation across distinct medical imaging domains. The implementation draws on the in-context task encoding and reference-guided decoding principles presented in Show and Segment: Universal Medical Image Segmentation via In-Context Learning (CVPR 2025), while introducing an independent dataset-specific EMA memory design.

This repository documents the implementation, experimental methodology, and quantitative and qualitative evaluation.

**An ongoing two-client prototype-sharing extension built on this framework is also documented below (see [Federated extension](#federated-extension); method and equations are documented there).**

## Predictions

The repository snapshot contains  four prediction montages, stored under `assets/`. Overlays use **green for ground truth** and **red for prediction**. 


### ACDC — cardiac foreground union

**All-axis comparison:** image, ground truth, one-support, three-support and EMA-memory predictions.

![ACDC all-axis predictions](assets/acdc_all_axes.png)

**Axis-2 cross-sectional views:** representative slices through the cardiac foreground.

![ACDC axis-2 predictions](assets/acdc_axis2_slices.png)

### MSD Brain Tumour — whole-tumour union

**All-axis comparison:** image, ground truth and the three inference strategies.

![Brain Tumour all-axis predictions](assets/msd_brain_tumor_all_axes.png)

**Axis-2 cross-sectional views:** representative slices through the whole-tumour foreground.

![Brain Tumour axis-2 predictions](assets/msd_brain_tumor_axis2_slices.png)

## Experiment

The current experiment is binary:

| Dataset | Binary foreground target |
|---|---|
| ACDC | Original labels 1 (RV), 2 (myocardium) and 3 (LV) are merged into one cardiac-foreground mask. |
| MSD Brain Tumour | All non-zero tumour labels are merged into one whole-tumour mask. |

Both datasets use a single input channel in this experiment. Results therefore measure the foreground unions; they do not measure individual cardiac structures or tumour subregions.


## Model

The implementation has four main stages:

1. **3D feature encoder.** A residual 3D U-Net-style encoder extracts volumetric features at five scales.
2. **Mask-guided task encoding.** A support mask is used to pool foreground features. A contextual branch fuses the mask with pixel-shuffled features, then learned query tokens attend to the contextual representation.
3. **Task-conditioned decoder.** Bidirectional cross-attention connects task tokens and query features. A residual decoder uses skip connections and FiLM feature modulation to produce a binary mask.
4. **Dataset-specific EMA task memory.** A persistent task embedding is maintained separately for each dataset and can be retrieved to segment a query without encoding a new labelled support image.

### Task embeddings

For the documented configuration, the encoder has 256 channels at its deepest stage. The task encoder produces one pooled foreground token and eight learned contextual tokens, giving a **9 × 256** task embedding per binary task.

The contextual branch uses the encoder's 16× spatial downsampling ratio. With 256 feature channels, the implementation computes:

```math
C_{\text{context}} =
\max\left(
1,
\left\lceil \frac{256}{16^3} \right\rceil
\right) = 1.
```

### EMA memory

The built-in `ClassMemoryBank` is keyed by integer IDs. Since both datasets use local foreground label `1` but represent different tasks, this experiment assigns separate memory keys:

| Dataset | Segmentation label | EMA memory key |
|---|---:|---:|
| ACDC cardiac foreground | 1 | 101 |
| MSD Brain Tumour whole tumour | 1 | 201 |

For task embedding \(E_t\), the memory update is:

```math
M_t = \mu M_{t-1} + (1-\mu)E_t,
\qquad \mu = 0.90.
```

Only detached support embeddings from training episodes update the bank, and only after a successful optimizer step. Validation and test do not update it. At bank-inference time, the runner retrieves the stored embedding associated with the selected dataset/task and sends it to the query decoder. This is support-free inference for **tasks already represented in the training memory**; it is not inference on an unseen task or unseen dataset.

## Evaluation protocol

The runner compares three inference strategies on the same sampled query episodes:

- **One support:** encode one labelled support volume and use its task embedding.
- **Context ensemble:** encode three distinct labelled support volumes and average their task embeddings.
- **EMA memory:** retrieve the persistent dataset-specific embedding without using a query-specific support.

Training uses one support and a different query subject per episode. ACDC frames are grouped by patient for splitting and support/query sampling, preventing different frames from one patient being treated as independent subjects. Validation episode sampling is fixed across checkpoints. Both datasets are used during training, so this experiment is a within-dataset held-out-subject evaluation, not unseen-dataset generalization.

The Dice metric is:

$$Dice(P, G) = \frac{2 |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$

where \(P\) is the predicted foreground and \(G\) is the ground-truth foreground. The reported macro score first averages episode Dice within each dataset and then gives the two datasets equal weight:

```math
D_{\text{macro}} =
\frac{D_{\text{ACDC}}+D_{\text{Brain Tumour}}}{2}.
```

This is an equal-weight average of dataset scores, not a pooled voxel-wise Dice.

## Training configuration

| Parameter | Documented value |
|---|---:|
| Volume size | 96 × 96 × 96 |
| Input channels | 1 |
| Base channels | 16 |
| Encoder downsampling stages | 4 (total factor 16) |
| Encoder channel widths | 16, 32, 64, 128, 256 |
| Learned contextual query tokens | 8 (+ 1 pooled foreground token) |
| Training iterations | 12,000 |
| Validation interval | 500 iterations |
| Validation episodes | 20 per dataset |
| Test episodes | 30 per dataset |
| Training supports / evaluation ensemble supports | 1 / 3 |
| Optimizer | LAMB |
| Initial learning rate | 1e-4 |
| LAMB betas / epsilon | (0.9, 0.999) / 1e-6 |
| Weight decay | 1e-5 |
| Gradient clipping | 1.0 maximum norm |
| Learning-rate schedule | Cosine annealing; minimum LR 1e-6 |
| Loss | Soft Dice loss + binary cross-entropy with logits |
| EMA momentum | **0.90** |
| Split | 70% train / 15% validation / 15% test |
| Split seed | 42 |
| Mixed precision | Enabled |

Images are resized to the target volume size using cubic interpolation; masks use nearest-neighbour interpolation. MRI intensities are percentile-clipped (1st–99th percentile), normalized, foreground-standardized and clipped to the configured range. No original image spacing target is specified in the documented run.

## Results
The framework achieved an **equal-weight macro validation Dice of 0.83** across ACDC cardiac MRI and MSD Brain Tumour MRI, based on the dataset-specific EMA memory-bank inference strategy.

### Validation

**Equal-weight macro validation Dice using EMA memory retrieval.** The selected checkpoint is chosen using this validation metric.

![Macro validation Dice](assets/validation_dice_macro.png)


## Repository layout

```text
.
├── README.md
├── .gitignore
├── assets/                          # One validation graph and four prediction montages
├── scripts/training/
│   └── train_iris_acdc_brain_binary_ensemble_memory_bank.py
└── src/iris/
    ├── data/                        # Volume I/O, preprocessing and dataset adapters
    ├── model/                       # Encoder, task encoder, decoder and EMA memory
    └── training/                    # LAMB optimizer and training utilities
```

The dataset files, model checkpoints and serialized memory-bank tensors are not included. The runner expects the complete `src/iris/` package and a local dataset root.

## Setup and run

Install a CUDA-compatible PyTorch build for your system first. Then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy nibabel matplotlib
```

`SimpleITK` is only needed if you load MHD/MHA files; the supplied ACDC and MSD loaders use NIfTI files. Place the datasets in the following layout (dataset contents are not included):

```text
datasets/
├── acdc/
└── msd_brain_tumor/
```

Run a short pipeline check:

```powershell
python -m py_compile scripts/training/train_iris_acdc_brain_binary_ensemble_memory_bank.py
python scripts/training/train_iris_acdc_brain_binary_ensemble_memory_bank.py `
  --dataset-root datasets `
  --datasets acdc msd_brain_tumor `
  --smoke-test `
  --resolution 32 `
  --eval-shots 3 `
  --memory-momentum 0.90 `
  --output-dir outputs/iris_ema_memory_smoke32
```

Reproduce the documented run:

```powershell
python scripts/training/train_iris_acdc_brain_binary_ensemble_memory_bank.py `
  --dataset-root datasets `
  --datasets acdc msd_brain_tumor `
  --iterations 12000 `
  --eval-every 500 `
  --val-episodes 20 `
  --test-episodes 30 `
  --eval-shots 3 `
  --resolution 96 `
  --memory-momentum 0.90 `
  --output-dir outputs/iris_acdc_brain_binary_ema_memory96_run12000
```

## Federated extension

The episodic model is extended to a two-client prototype-sharing setup. A two-client configuration is used first to verify the communication loop and observe how partial task overlap, split imbalance, and modality differences (MRI-only vs. MRI and CT) affect label-free clustering before scaling to more clients.
For this experiment, the model is configured for **128 × 128 × 128** (single channel), with the same 9 × 256 task embeddings. ACDC training patients are randomly split by patient ID (seed 42): 75% are assigned to Client A and the remaining 25% to Client B. Client A also uses MSD Brain Tumour, while Client B uses MSD Spleen. Each client maintains its own model, optimizer and EMA memory. 

```mermaid
flowchart LR
    subgraph A["Client A"]
        A1["75% ACDC + MSD Brain Tumour"] --> A2["Local model and EMA memory"]
    end
    subgraph B["Client B"]
        B1["25% ACDC + MSD Spleen"] --> B2["Local model and EMA memory"]
    end
    A2 -->|"Prototypes, Q and S"| S["Server: weighted K-means"]
    B2 -->|"Prototypes, Q and S"| S
    S -->|"3 normalized centroids"| A2
    S -->|"3 normalized centroids"| B2
```

For each client-task memory entry $E_{c,k}\in\mathbb{R}^{9\times256}$, the server receives a normalized 256-dimensional prototype:

```math
p_{c,k}=\frac{\frac{1}{9}\sum_{j=1}^{9}E_{c,k,j}}
{\left\|\frac{1}{9}\sum_{j=1}^{9}E_{c,k,j}\right\|_2+\epsilon}.
```

Each prototype is accompanied by $Q_i$, the foreground-voxel count accumulated during the communication interval, and $S_i$, the mean predicted foreground probability over ground-truth foreground voxels. For the $M$ pooled prototypes, the server combines their normalized statistics to obtain sample weights and performs weighted K-means with three clusters:

```math
\widetilde{w}_i=M\left(\frac{Q_i}{\sum_{\ell=1}^{M}Q_\ell+\epsilon}+\frac{S_i}{\sum_{\ell=1}^{M}S_\ell+\epsilon}\right),\qquad
w_i=\frac{\widetilde{w}_i}{\sum_{q=1}^{M}\widetilde{w}_q}.
```

Writing the pooled prototype vectors as $p_i$, weighted K-means with three clusters minimizes:

```math
\min_{\{\mu_r\}_{r=1}^{3}}\sum_{i=1}^{M}w_i\min_{r\in\{1,2,3\}}\left\|p_i-\mu_r\right\|_2^2.
```

The resulting centroids $c_r$ are L2-normalized and returned to both clients. For each local task token $x_j$, the alignment term uses its softmax similarity to the centroids, weighted by the distance to each centroid:

```math
\hat{x}_j=\frac{x_j}{\max(\|x_j\|_2,\epsilon)},\qquad
a_{jr}=\frac{\exp(\hat{x}_j^\top c_r/\tau)}{\sum_{s=1}^{3}\exp(\hat{x}_j^\top c_s/\tau)}.
```

```math
\mathcal{L}_{\mathrm{align}}=\frac{1}{9}\sum_{j=1}^{9}\sum_{r=1}^{3}a_{jr}\left\|\hat{x}_j-c_r\right\|_2,\qquad
\mathcal{L}_{\mathrm{client}}=\mathcal{L}_{\mathrm{seg}}+0.1\mathcal{L}_{\mathrm{align}},\quad \tau=0.1.
```

Communication begins after 1,000 local iterations and repeats every 500 iterations. The server clusters prototype vectors without using task IDs as clustering features; $Q_i$ and $S_i$ are used as sample weights. Raw scans, masks and model weights remain local. **The federated extension is documented at the methodological level and its training scripts are not included in this public release.**


## Reference

Gao, Y. et al. (2025). *Show and Segment: Universal Medical Image Segmentation via In-Context Learning*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

- [arXiv:2503.19359](https://arxiv.org/abs/2503.19359)
