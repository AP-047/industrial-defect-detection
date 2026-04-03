# Industrial Defect Detection using Deep Learning and Model Explainability

A computer vision project for binary defect classification on steel surface images.

This project uses transfer learning with ResNet18 and Grad-CAM for visual explanation.

GitHub About (one-line): Transfer learning based industrial surface defect detection using ResNet18 with Grad-CAM interpretability on the NEU Surface Defect dataset.

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

```bash
python src/gradcam.py --image-path data/val/defect/pitted_surface_247.jpg --model-path outputs/model.pth
```

## Project files

- `src/dataset.py`: data loading and transforms
- `src/model.py`: transfer learning model definition
- `src/train.py`: training loop and model saving
- `src/gradcam.py`: Grad-CAM visualization

## Results snapshot

- Validation accuracy: add your best score from training logs
- Qualitative result: Grad-CAM highlights defect regions on validation samples

## Notes

- Focused on practical deep learning workflow: data loading, transfer learning, training, and model explainability.
- Training supports a fixed random seed (`--seed`) for more reproducible results.