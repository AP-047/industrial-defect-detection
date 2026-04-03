import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx):
        activations = None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        handle = self.target_layer.register_forward_hook(forward_hook)

        output = self.model(input_tensor)

        handle.remove()

        # get target class score
        score = output[:, class_idx]

        # compute gradients
        grads = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            grad_outputs=torch.ones_like(score),
            retain_graph=True
        )[0]

        activations = activations.detach().cpu().numpy()[0]
        grads = grads.detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam

def visualize(image_path):
    model = get_model()
    model.load_state_dict(torch.load("outputs/model.pth"))
    model.to(DEVICE)
    model.eval()

    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)

    transform = get_transforms()

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))

    input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    _, pred = torch.max(output, 1)

    cam = gradcam.generate(input_tensor, pred.item())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + np.float32(rgb) / 255
    overlay = overlay / np.max(overlay)

    plt.imshow(overlay)
    plt.title(f"Prediction: {pred.item()}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    visualize("data/val/defect/scratches_247.jpg")