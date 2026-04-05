# Industrial Defect Detection using Deep Learning and Model Explainability

A deep learning-based computer vision system for automated surface defect detection in industrial materials.

This project leverages transfer learning with ResNet18 and integrates Grad-CAM to provide visual explanations of model predictions.

## Overview

- Task: Binary classification (`normal` vs `defect`)
- Backbone: ResNet18 (transfer learning)
- Explainability: Grad-CAM
- Output: trained weights at `outputs/model.pth`

## Dataset

- Source: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- In this repo, data is already arranged in class folders:

```text
data/
	train/
		normal/
		defect/
	val/
		normal/
		defect/
```
Note: Since the dataset does not contain a true "normal" class, the "patches" category was treated as normal, while all other defect categories were grouped as defect.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train model:

```bash
python src/train.py --seed 42
```

3. Run Grad-CAM visualization:
Edit the image path in `src/gradcam.py` to point to a validation image, then run:
```bash
python src/gradcam.py
```

## Project structure

- `src/dataset.py`: data loading and transforms
- `src/model.py`: transfer learning model definition
- `src/train.py`: training loop and model saving
- `src/gradcam.py`: Grad-CAM visualization

## Results snapshot

- Validation accuracy: ~100% on validation split (limited dataset)
- Qualitative result: Grad-CAM highlights defect regions on validation samples

### Example Output

Below is a Grad-CAM visualization showing model attention:

![GradCAM Example](./outputs/example-1.png)

## Notes

- Focused on practical deep learning workflow: data loading, transfer learning, training, and model explainability.
- Training supports a fixed random seed (`--seed`) for more reproducible results.