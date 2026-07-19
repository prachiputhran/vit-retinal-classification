import torch
import timm
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = timm.create_model("vit_base_patch16_224", pretrained=False,
                           num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
    print(f"Prediction : {CLASS_NAMES[pred]}")
    print(f"Confidence : {probs[pred]*100:.1f}%")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"  {cls}: {p*100:.1f}%")

# Usage: predict("path/to/retinal_image.jpg")