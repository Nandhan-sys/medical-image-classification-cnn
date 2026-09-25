# Medical Image CNN — Pneumonia Detection from Chest X-Rays

A PyTorch project that trains a custom convolutional neural network to classify chest X-ray images as **NORMAL** or **PNEUMONIA**, using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## Overview

The notebook builds two versions of the same CNN architecture and compares them:

- **Experiment A — Plain CNN**: trained with standard cross-entropy loss.
- **Experiment B — Weighted CNN**: trained with class-weighted cross-entropy loss to counter class imbalance (the dataset has roughly 3x more PNEUMONIA images than NORMAL in training).

Both models are evaluated on a held-out validation set, and the final model is evaluated once on a locked test set that is never touched during training or tuning.

## Dataset

Downloaded automatically via `kagglehub`:

```
kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train (original) | 1,341 | 3,875 | 5,216 |
| Val (original) | 8 | 8 | 16 |
| Test (locked) | 234 | 390 | 624 |

The original validation split (16 images) is too small to be reliable, so it's discarded. Instead, the notebook creates a new **stratified 90/10 split** of the training data (train/val), preserving the NORMAL:PNEUMONIA ratio. The original test set of 624 images stays locked and untouched until final evaluation.

## Pipeline

1. **Setup** — fixed random seed (42) for reproducibility; GPU detection.
2. **Download dataset** via `kagglehub`.
3. **Explore raw folders** to confirm class counts per split.
4. **Transforms** — resize to 224x224, normalize with ImageNet mean/std; training set also gets random horizontal flip augmentation.
5. **Stratified re-split** — training folder is loaded twice (once augmented, once clean), then split 90/10 by index using `sklearn.train_test_split` with `stratify`, so the new validation set uses un-augmented images.
6. **DataLoaders** — batch size 32.
7. **Data sanity check** — visualize a sample image.
8. **Model** — a small custom CNN:
   - Conv2d(3→10, 3x3) → ReLU → MaxPool(2x2)
   - Conv2d(10→20, 3x3) → ReLU
   - Flatten → Linear(250880→2)
9. **Class weights** — recomputed from the new train split (not the original) to correctly counter imbalance.
10. **Training/evaluation utilities** — shared `train_model()` and `evaluate()` functions used by both experiments, plus a `metrics_report()` helper that prints accuracy/precision/recall/F1 and plots a confusion matrix heatmap.
11. **Experiment A** — plain CNN, 3 epochs, Adam (lr=0.001), unweighted `CrossEntropyLoss`.
12. **Experiment B** — same architecture and hyperparameters, but `CrossEntropyLoss` weighted by class frequency.
13. **Model checkpoints** — both models saved as `cnn_plain.pth` and `cnn_weighted.pth`.
14. **Final locked-test-set evaluation** — the plain CNN is evaluated once on the untouched 624-image test set.

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
kagglehub
```

## Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib seaborn kagglehub
   ```
2. Open and run `Medical_Image_CNN_v3.ipynb` top to bottom in Jupyter, VS Code, or Google Colab.
3. The dataset downloads automatically on first run via `kagglehub` (a Kaggle account/API token may be required).
4. Trained weights are saved to `cnn_plain.pth` and `cnn_weighted.pth` in the working directory.

## Notes

- Results are seeded (`seed=42`) for reproducibility across runs.
- The locked test set is only used for a final, single evaluation pass to avoid test-set leakage during model iteration.

## Disclaimer

This project is for educational/experimental purposes only and is **not** a validated diagnostic tool. It should not be used for real clinical decision-making.
