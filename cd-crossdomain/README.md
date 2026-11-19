# Pseudo-Change Is Not Separable — code release

Code for the paper:

> **Pseudo-Change Is Not Separable: A Diagnostic Study of Cross-Domain Building
> Change Detection**

The repository contains the training pipeline, every evaluation script used in
the paper, and the analysis scripts that produce each table and figure. It is
organised so that each section of the paper can be reproduced independently.

## What this paper claims

Across three building change detection datasets, pseudo-change accounts for
40.7% and 45.0% of false positives on the two target domains but only 4.0% on
the source. Four suppression mechanisms were implemented and all of them cost
more than the total benefit they could recover (4.18 to 7.03 points against a
perfect-removal ceiling of 7.07). A feature-space analysis explains why: in the
fused representation, pseudo-change occupies a subregion of the true-change
distribution, which imposes a lower bound of roughly 38% false suppression on
any discriminator operating there.

The point of releasing this code is reproducibility of a **negative** result.
If you suspect a mechanism was simply mistuned, the scripts here let you rerun
it with different weights, perturbation ranges or thresholds.

## Requirements

```
Python 3.11
torch 2.7.1+cu118   (a newer CUDA build works too)
torchvision, numpy<2, opencv-python, matplotlib, einops, pymupdf
```

Install:

```bash
pip install -r requirements.txt
```

All experiments in the paper were run on a single 4 GB GTX 1650; one training
run takes about 9 minutes for 30 epochs.

## Data

Download the three public datasets and place them under `data/`:

| Dataset   | Source |
|-----------|--------|
| LEVIR-CD  | https://justchenhao.github.io/LEVIR/ |
| WHU-CD    | (author-provided; see the original paper) |
| S2Looking | https://github.com/S2Looking/Dataset |

Then build the patch sets:

```bash
python build_patches.py        # LEVIR-CD  -> data/patches
python build_patches_whu.py    # WHU-CD    -> data/patches_whu
python build_patches_s2l.py    # S2Looking -> data/patches_s2l
```

Each patch set contains `train/`, `val/`, `test/` with `A/`, `B/`, `L/`
subfolders of `.npy` files (uint8 images, binary labels).

## Reproducing the paper

### Train the source model (§3, §4.1)

```bash
python train.py --ratio 0.1 --seed 0 --epochs 30
python train.py --ratio 0.1 --seed 1
python train.py --ratio 0.1 --seed 2
```

Weights are written to `runs/base_seed{0,1,2}/best.pt`.

For the fully supervised comparison of §4.3, use `--ratio 0.4`.

### Cross-domain evaluation (§4)

```bash
python bench_protocol.py                 # all three domains, fixed threshold 0.5
python evaluate.py --splits test --root data/patches_whu --ckpt runs/base_seed0/best.pt
python probe_norm.py --ckpt runs/base_seed0/best.pt --n-images 200   # §4.4
```

### Error decomposition (§5)

```bash
python error_decomp.py --ckpt runs/base_seed0/best.pt --root data/patches      --split test
python error_decomp.py --ckpt runs/base_seed0/best.pt --root data/patches_whu  --split test
python error_decomp.py --ckpt runs/base_seed0/best.pt --root data/patches_s2l  --split test
```

Output separates false positives into pseudo-change (zero-change patches),
border (<= 3 px) and far (> 3 px), and reports the F1 that would result from
removing each class.

### Suppression mechanisms (§6)

```bash
# global / localized photometric consistency, zero-change supervision
python train.py --cons-weight 1.0 --cons-scope global  --seed 0
python train.py --cons-weight 1.0 --cons-scope local   --seed 0
python train.py --zero-per-batch 2 --seed 0

# area post-processing sweep
python eval_area_filter.py --ckpt runs/base_seed0/best.pt --root data/patches_whu

# hard-negative mining (reported as a rejected route)
python mine_hard_negatives.py
```

Aggregate multi-seed results:

```bash
python aggregate.py --group "global=runs/cons_global_seed*"
```

### Feature separability (§7)

```bash
python feature_separable.py --root data/patches_whu --split test
```

Produces nearest-centroid accuracy for pseudo-change versus true positives and
a PCA projection under `vis/`.

### Cross-architecture check (§5.2, §7.2)

These scripts use the official BIT implementation. Clone it first:

```bash
git clone https://github.com/justchenhao/BIT_CD.git third_party/BIT_CD
```

Then place the official LEVIR-CD checkpoint at
`data/LEVIR-CD/BIT_LEVIR/best_ckpt.pt` and run:

```bash
python bit_error_decomp.py      --root data/patches_whu --split test
python bit_error_decomp.py      --root data/patches      --split test
python bit_feature_separable.py --root data/patches_whu --split test
```

Note: BIT is written for an older torchvision. Both scripts install a small
compatibility shim for `torchvision.models.utils` before importing it.

### Figures

```bash
python make_figures.py
python make_figure_qualitative.py
```

## Repository layout

```
train.py                 CDNet, loss, training loop
data.py                  PatchDataset, loaders, bucketing
make_splits.py           train/val/test split construction
build_patches*.py        dataset -> 256x256 patch conversion

evaluate.py              single-checkpoint evaluation
bench_protocol.py        unified three-domain evaluation (fixed threshold)
aggregate.py             mean/std over seeds
eval_crossdomain.py      per-domain sweep

error_decomp.py          pseudo-change / border / far decomposition   (§5)
eval_area_filter.py      area post-processing sweep                  (§6)
mine_hard_negatives.py   hard-negative mining (rejected)             (§6)
probe_norm.py            inference-time normalization comparison    (§4.4)
analyze_cc.py            connected-component size statistics        (§5.3)
feature_separable.py     nearest-centroid separability + PCA         (§7)

bit_error_decomp.py      same decomposition with BIT                 (§5.2)
bit_feature_separable.py same separability with BIT                  (§7.2)

make_figures.py          figures 1-3, 5
make_figure_qualitative.py  figure 4

tools/                   dataset inspection and sanity checks
```

## Notes on the protocol

* Fixed threshold 0.5 everywhere. The per-domain threshold search reported as
  an upper bound is never used as a headline number.
* Three seeds per configuration; the baseline standard deviation on WHU-CD is
  0.0019, which is why differences of 0.4 points are treated as real.
* Predictions are compared against ground truth on full test sets; S2Looking
  uses all 1000 pairs.

## Contact

Questions about the code: zhangripei@foxmail.com
