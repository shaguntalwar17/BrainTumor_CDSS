from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.classification.train import build_model


def find_last_conv(module: torch.nn.Module):
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise RuntimeError("No Conv2d layer found for Grad-CAM.")


def generate_gradcam(checkpoint_path: str, image_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    classes = ckpt["classes"]
    model = build_model(model_name, num_classes=len(classes), dropout=0.35)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layer = find_last_conv(model)
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    pred_class = torch.argmax(logits, dim=1).item()

    model.zero_grad(set_to_none=True)
    logits[:, pred_class].backward()

    acts = activations["value"][0]
    grads = gradients["value"][0]

    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_np = cam.cpu().numpy()

    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (224, 224))
    heat = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)

    out = np.concatenate([orig, heat, overlay], axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, out)

    h1.remove()
    h2.remove()

    print(f"Predicted class: {classes[pred_class]} | saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization")
    parser.add_argument("--checkpoint", type=str, default="ml/artifacts/classification/best_classification.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="sample_outputs/gradcam_result.png")
    args = parser.parse_args()

    generate_gradcam(args.checkpoint, args.image, args.output)
