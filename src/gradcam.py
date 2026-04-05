import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_transforms


# Keep device selection centralized so inference works on CPU or GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GradCAM:
    """Minimal Grad-CAM implementation for visualizing model focus areas."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx):
        # Store feature maps from the chosen layer during a forward pass.
        activations = None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        # Hook into the target layer so we can reuse its activations later.
        handle = self.target_layer.register_forward_hook(forward_hook)

        # Run a forward pass to get both predictions and captured activations.
        output = self.model(input_tensor)

        handle.remove()

        # Use the score for the predicted class as the gradient target.
        score = output[:, class_idx]

        # Backpropagate the class score into the target activations.
        grads = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            grad_outputs=torch.ones_like(score),
            retain_graph=True
        )[0]

        # Convert tensors to NumPy for the weighted heatmap calculation.
        activations = activations.detach().cpu().numpy()[0]
        grads = grads.detach().cpu().numpy()[0]

        # Channel-wise mean gradients act as importance weights.
        weights = np.mean(grads, axis=(1, 2))

        # Combine weighted feature maps into a coarse class activation map.
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU keeps only positive evidence for the predicted class.
        cam = np.maximum(cam, 0)
        # Resize to the original image resolution for visualization.
        cam = cv2.resize(cam, (224, 224))

        # Normalize to [0, 1] so it can be converted into a heatmap.
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam


def visualize(image_path, model_path):
    # Recreate the same architecture used during training.
    model = get_model()
    # Load weights on the current device so the script works on CPU-only machines.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Use the last convolutional block for Grad-CAM, since it carries high-level features.
    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)

    # Use the same preprocessing pipeline as training.
    transform = get_transforms()

    # Read the selected image and convert it into the format expected by torchvision.
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))

    # Add a batch dimension because the model expects batched input.
    input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    # Predict the class and use that class index for the Grad-CAM target.
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

    cam = gradcam.generate(input_tensor, pred.item())

    # Convert the normalized CAM into a heatmap and blend it with the original image.
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + np.float32(rgb) / 255
    overlay = overlay / np.max(overlay)

    # Display the final explanation overlay.
    plt.imshow(overlay)
    plt.title(f"Prediction: {pred.item()}")
    plt.axis("off")
    plt.show()


def parse_args():
    """Parse command-line arguments for a flexible Grad-CAM demo."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization.")
    parser.add_argument(
        "--image-path",
        default="data/val/defect/pitted_surface_247.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/model.pth",
        help="Path to trained model weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.image_path, args.model_path)