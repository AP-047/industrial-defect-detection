# Industrial Defect Detection

A simple deep learning project for binary defect classification on steel surface images.

This project uses transfer learning with ResNet18 and Grad-CAM for visual explanation.

## What this project does

- Classifies images into 2 classes: `normal` and `defect`
- Trains a ResNet18-based model on NEU Surface Defect images
- Saves the trained model to `outputs/model.pth`
- Generates Grad-CAM heatmaps to show where the model is focusing

## Dataset

- Source: NEU Surface Defect Database
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

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train model:

```bash
python src/train.py
```

4. Run Grad-CAM visualization:

```bash
python src/gradcam.py
```

## Files (quick overview)

- `src/dataset.py`: Loads images and applies transforms
- `src/model.py`: Defines transfer learning model (ResNet18)
- `src/train.py`: Trains model and saves weights
- `src/gradcam.py`: Visualizes model attention using Grad-CAM

## Output

- Trained model: `outputs/model.pth`
- Example analysis images: `data/Analysis/`

## Notes

- This is a student project focused on clear and simple implementation.
- Current target task is binary classification (`normal` vs `defect`).