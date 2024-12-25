import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn

# === MODEL CLASS ===
class EnhancedPneumoniaModel(nn.Module):
    def __init__(self, pretrained=False):
        super(EnhancedPneumoniaModel, self).__init__()
        self.backbone = efficientnet_v2_s(weights=None if not pretrained else EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# === PREDICTION FUNCTION ===
def predict_image(model, image_path, device):
    # Transforms for the input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor).squeeze()
        prediction = torch.sigmoid(output).item()
    return prediction

# === GUI ===
class PneumoniaPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Prediction App")
        self.root.geometry("600x450")
        self.root.configure(bg="#f4f4f4")

        # Title Label
        self.title_label = tk.Label(
            root, text="Pneumonia Detection", font=("Helvetica", 20, "bold"), bg="#f4f4f4", fg="#333"
        )
        self.title_label.pack(pady=20)

        # Load Model Button
        self.load_model_button = tk.Button(
            root, text="Load Model", command=self.load_model, font=("Helvetica", 12), bg="#4CAF50", fg="white"
        )
        self.load_model_button.pack(pady=10)

        # Select Image Button
        self.select_image_button = tk.Button(
            root, text="Select Image", command=self.select_image, font=("Helvetica", 12), bg="#2196F3", fg="white"
        )
        self.select_image_button.pack(pady=10)

        # Label to display selected image
        self.image_label = tk.Label(root, bg="#f4f4f4")
        self.image_label.pack(pady=10)

        # Prediction Label
        self.prediction_label = tk.Label(
            root, text="Prediction: ", font=("Helvetica", 16), bg="#f4f4f4", fg="#333"
        )
        self.prediction_label.pack(pady=20)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File", filetypes=[("PyTorch Model", "*.pth")]
        )
        if model_path:
            self.model = EnhancedPneumoniaModel(pretrained=False).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            messagebox.showinfo(
                "Model Loaded", "The model has been successfully loaded and is ready to use!"
            )

    def select_image(self):
        if self.model is None:
            messagebox.showwarning("No Model Loaded", "Please load a model first!")
            return

        image_path = filedialog.askopenfilename(
            title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if image_path:
            # Display the image
            image = Image.open(image_path)
            image.thumbnail((300, 300))
            image = ImageTk.PhotoImage(image)
            self.image_label.config(image=image, text="")
            self.image_label.image = image

            # Predict
            prediction = predict_image(self.model, image_path, self.device)
            prediction_status = "High likelihood of Pneumonia" if prediction > 0.5 else "Low likelihood of Pneumonia"
            color = "red" if prediction > 0.5 else "green"
            self.prediction_label.config(
                text=f"Prediction: {prediction:.4f} ({prediction_status})", fg=color
            )

# === MAIN ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaPredictionApp(root)
    root.mainloop()
