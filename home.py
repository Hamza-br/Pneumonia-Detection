import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from torchvision import transforms
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import os

class EnhancedPneumoniaModel(nn.Module):
    def __init__(self, pretrained=True):
        super(EnhancedPneumoniaModel, self).__init__()
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.features(torch.rand(1, 3, 224, 224)).flatten(1).shape[1]
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features + 3, 512),  # Include metadata features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, metadata):
        features = self.backbone.features(x)
        features = torch.flatten(features, 1)
        combined = torch.cat((features, metadata), dim=1)
        return self.backbone.classifier(combined)

# === PREDICTION FUNCTION ===
def predict_image(model, image_path, metadata, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    metadata_tensor = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor, metadata_tensor).squeeze()
        prediction = torch.sigmoid(output).item()
    return prediction


class PneumoniaPredictionApp:    
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detection System")
        self.root.configure(bg='#f0f0f0')  # Light gray background
        self.setup_styles()
        self.create_main_frame()
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path = None
        self.current_image = None
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure common colors
        PRIMARY_COLOR = '#2196F3'  # Blue
        SUCCESS_COLOR = '#4CAF50'  # Green
        BG_COLOR = '#f0f0f0'       # Light gray
        
        # Button styles
        style.configure('Blue.TButton',
                       background=PRIMARY_COLOR,
                       foreground='white',
                       padding=(0, 10),
                       font=('Segoe UI', 10))
        style.configure('Green.TButton',
                       background=SUCCESS_COLOR,
                       foreground='white',
                       padding=(0, 10),
                       font=('Segoe UI', 10))
        
        # Frame styles
        style.configure('Main.TFrame', background=BG_COLOR)
        style.configure('Controls.TLabelframe', 
                       background=BG_COLOR,
                       font=('Segoe UI', 10, 'bold'))
        style.configure('Controls.TLabelframe.Label', 
                       background=BG_COLOR,
                       font=('Segoe UI', 10, 'bold'))
                       
        # Label styles
        style.configure('Info.TLabel',
                       background=BG_COLOR,
                       font=('Segoe UI', 10))
        style.configure('Status.TLabel',
                       background=BG_COLOR,
                       font=('Segoe UI', 9),
                       foreground='#666666')
                       
        # Result box styles
        style.configure('Result.TFrame',
                       background='white',
                       relief='solid',
                       borderwidth=1)
        style.configure('ResultTitle.TLabel',
                       background='white',
                       font=('Segoe UI', 12, 'bold'),
                       foreground='#000080')  # Dark blue
        style.configure('ResultValue.TLabel',
                       background='white',
                       font=('Segoe UI', 24, 'bold'),
                       foreground='#000080')  # Dark blue
        style.configure('ResultText.TLabel',
                       background='white',
                       font=('Segoe UI', 12),
                       foreground='#000080')  # Dark blue
                       
    def create_main_frame(self):
        main_frame = ttk.Frame(self.root, style='Main.TFrame', padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create left and right panels
        self.create_control_panel(main_frame)
        self.create_image_panel(main_frame)
        
    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controls", style='Controls.TLabelframe', padding=15)
        control_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Load Model button
        load_btn = ttk.Button(control_frame, text="Load Model", 
                            style='Blue.TButton', command=self.load_model)
        load_btn.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        # Select Image button
        select_btn = ttk.Button(control_frame, text="Select Image",
                              style='Blue.TButton', command=self.select_image)
        select_btn.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        # Patient Information section
        info_frame = ttk.LabelFrame(control_frame, text="Patient Information",
                                  style='Controls.TLabelframe', padding=10)
        info_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))
        
        # Age input
        ttk.Label(info_frame, text="Age:", style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.age_var = tk.StringVar()
        self.age_entry = ttk.Entry(info_frame, textvariable=self.age_var, width=15)
        self.age_entry.grid(row=0, column=1, sticky='w', padx=(5, 0), pady=5)
        self.age_var.trace('w', lambda *args: self.validate_age())
        
        # Sex selection
        ttk.Label(info_frame, text="Sex:", style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        self.sex_var = tk.StringVar(value="M")
        ttk.Radiobutton(info_frame, text="Male", variable=self.sex_var,
                       value="M").grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(info_frame, text="Female", variable=self.sex_var,
                       value="F").grid(row=1, column=2, sticky='w')
        
        # Position selection
        ttk.Label(info_frame, text="Position:", style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        self.position_var = tk.StringVar(value="AP")
        ttk.Radiobutton(info_frame, text="AP", variable=self.position_var,
                       value="AP").grid(row=2, column=1, sticky='w')
        ttk.Radiobutton(info_frame, text="PA", variable=self.position_var,
                       value="PA").grid(row=2, column=2, sticky='w')
        
        # Predict button
        predict_btn = ttk.Button(control_frame, text="Predict",
                               style='Green.TButton', command=self.predict)
        predict_btn.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                               style='Status.TLabel', wraplength=250)
        status_label.grid(row=4, column=0, sticky='w', pady=(0, 15))
        
        # Result box
        self.result_frame = ttk.Frame(control_frame, style='Result.TFrame')
        self.result_frame.grid(row=5, column=0, sticky='ew')
        self.result_frame.grid_remove()  # Hide initially
        
        # Configure result box layout
        self.result_frame.columnconfigure(0, weight=1)
        
        # Result title
        self.result_title = ttk.Label(self.result_frame, text="Result",
                                    style='ResultTitle.TLabel', anchor='center')
        self.result_title.grid(row=0, column=0, pady=(10, 5))
        
        # Result value (percentage)
        self.result_value = ttk.Label(self.result_frame, style='ResultValue.TLabel',
                                    anchor='center')
        self.result_value.grid(row=1, column=0, pady=5)
        
        # Result text (likelihood)
        self.result_text = ttk.Label(self.result_frame, style='ResultText.TLabel',
                                   anchor='center')
        self.result_text.grid(row=2, column=0, pady=(5, 10))
        
    def create_image_panel(self, parent):
        image_frame = ttk.LabelFrame(parent, text="X-Ray Image",
                                   style='Controls.TLabelframe', padding=10)
        image_frame.grid(row=0, column=1, sticky='nsew')
        
        # Configure grid weights
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(image_frame, bg='black', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL,
                                  command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL,
                                  command=self.canvas.xview)
        
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set,
                            yscrollcommand=v_scrollbar.set)
        
    def validate_age(self):
        try:
            age = float(self.age_var.get())
            if age < 0 or age > 120:
                self.age_entry.configure(style='Error.TEntry')
                return False
            self.age_entry.configure(style='TEntry')
            return True
        except ValueError:
            if self.age_var.get() != "":
                self.age_entry.configure(style='Error.TEntry')
                return False
            return True
            
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if model_path:
            try:
                self.model = EnhancedPneumoniaModel(pretrained=False).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()  # Set to evaluation mode
                self.status_var.set(f"Model loaded: {os.path.basename(model_path)}")
                self.result_frame.grid_remove()  # Hide result box when loading new model
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.status_var.set("Error loading model")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if self.image_path:
            self.result_frame.grid_remove()  # Hide result box when selecting new image
            self.display_image()
            
    def display_image(self):
        if self.image_path:
            # Clear previous image
            self.canvas.delete("all")
            
            # Load and resize image
            img = Image.open(self.image_path)
            aspect_ratio = img.width / img.height
            
            # Resize maintaining aspect ratio
            canvas_width = 500
            canvas_height = int(canvas_width / aspect_ratio)
            
            img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)}")

    def predict(self):
        if not self.validate_inputs():
            return
            
        try:
            metadata = [
                float(self.age_var.get()),
                1.0 if self.sex_var.get() == 'M' else 0.0,
                1.0 if self.position_var.get() == 'AP' else 0.0
            ]
            
            prediction = predict_image(self.model, self.image_path, metadata, self.device)
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            result = "High" if prediction > 0.5 else "Low"
            self.result_value.configure(text=f"{int(confidence * 100)}%\n{result} likelihood of pneumonia")
            self.result_frame.grid()  # Show result box
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
            self.result_frame.grid_remove()  # Hide result box on error
            
    def validate_inputs(self):
        if not self.model:
            messagebox.showwarning("Error", "Please load a model first")
            return False
        if not self.image_path:
            messagebox.showwarning("Error", "Please select an image first")
            return False
        if not self.validate_age():
            messagebox.showwarning("Error", "Please enter a valid age (0-120)")
            return False
        return True
    
    
if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaPredictionApp(root)
    root.mainloop()
