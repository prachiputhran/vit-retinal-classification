import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (cohen_kappa_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_data   = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)

model = timm.create_model("vit_base_patch16_224", pretrained=False,
                           num_classes=len(test_data.classes))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
print(f"Test Quadratic Kappa: {kappa:.4f}\n")
print(classification_report(all_labels, all_preds,
                             target_names=test_data.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=test_data.classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — ViT Retinal Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")