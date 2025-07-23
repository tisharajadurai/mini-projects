import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# ------------------- Settings -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Sickle']

# ------------------- Define Model -------------------
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("sickle_resnet50.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------- GUI -------------------
class SickleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sickle Cell Detection with ResNet50")
        self.geometry("700x500")

        self.label = ctk.CTkLabel(self, text="Upload a Blood Cell Image", font=("Arial", 18))
        self.label.pack(pady=20)

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)

        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.detect_button = ctk.CTkButton(self, text="Detect", command=self.detect_cell)
        self.detect_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

        self.file_path = None

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            img = Image.open(self.file_path).convert("RGB")
            img_resized = img.resize((300, 300))
            img_display = ImageTk.PhotoImage(img_resized)
            self.image_label.configure(image=img_display)
            self.image_label.image = img_display
            self.result_label.configure(text="")

    def detect_cell(self):
        if self.file_path is None:
            self.result_label.configure(text="No image selected!")
            return

        # Load and preprocess image
        image = Image.open(self.file_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_label = CLASS_NAMES[predicted.item()]

        self.result_label.configure(text=f"Prediction: {class_label}", text_color="green" if class_label == "Normal" else "red")

# ------------------- Run App -------------------
if __name__ == "__main__":
    app = SickleApp()
    app.mainloop()


