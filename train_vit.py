import os
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
val_data   = datasets.ImageFolder("dataset/val",   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False, num_workers=2)

# Model — freeze early blocks, fine-tune last 4
model = timm.create_model("vit_base_patch16_224", pretrained=True,
                           num_classes=len(train_data.classes))
for name, param in model.named_parameters():
    if "blocks.0" in name or "blocks.1" in name or \
       "blocks.2" in name or "blocks.3" in name:
        param.requires_grad = False
model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss_fn = torch.nn.CrossEntropyLoss()

best_kappa = -1

for epoch in range(20):
    # --- Train ---
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validate ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} val"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Kappa: {kappa:.4f}")

    scheduler.step()

    if kappa > best_kappa:
        best_kappa = kappa
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  Saved best model (kappa={kappa:.4f})")

print(f"\nBest Quadratic Kappa: {best_kappa:.4f}")