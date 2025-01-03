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
                       
        # Result styles
        style.configure('ResultTitle.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       foreground='#000080')
        style.configure('ResultValue.TLabel',
                       font=('Segoe UI', 24, 'bold'))
        style.configure('ResultText.TLabel',
                       font=('Segoe UI', 10),
                       foreground='#333333')
                       
    def create_main_frame(self):
        main_frame = ttk.Frame(self.root, style='Main.TFrame', padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)  # Give more weight to the image panel
        main_frame.columnconfigure(2, weight=1)  # Add column for results
        main_frame.rowconfigure(0, weight=1)
        
        # Create all panels
        self.create_control_panel(main_frame)
        self.create_image_panel(main_frame)
        self.create_results_panel(main_frame)
        
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
        
    def _on_frame_configure(self, event=None):
        """Handle resize events"""
        if hasattr(self, 'original_image'):
            self.display_image()
            
    def create_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", 
                                     style='Controls.TLabelframe', padding=15)
        results_frame.grid(row=0, column=2, sticky='nsew', padx=(10, 0))
        
        # Configure column weight
        results_frame.columnconfigure(0, weight=1)
        
        # Initial message
        self.no_results_label = ttk.Label(results_frame, 
                                        text="No analysis results yet.\nUse the controls on the left to load an image and make a prediction.",
                                        style='Status.TLabel',
                                        justify='center',
                                        wraplength=200)
        self.no_results_label.grid(row=0, column=0, pady=20)
        
        # Create result widgets (initially hidden)
        self.result_widgets_frame = ttk.Frame(results_frame)
        self.result_widgets_frame.grid(row=0, column=0, sticky='nsew')
        self.result_widgets_frame.grid_remove()  # Hide initially
        
        # Prediction probability
        self.probability_label = ttk.Label(self.result_widgets_frame,
                                         text="Prediction Probability",
                                         style='ResultTitle.TLabel')
        self.probability_label.grid(row=0, column=0, pady=(0, 5))
        
        self.probability_value = ttk.Label(self.result_widgets_frame,
                                         text="",
                                         style='ResultValue.TLabel')
        self.probability_value.grid(row=1, column=0, pady=(0, 15))
        
        # Severity assessment
        self.severity_label = ttk.Label(self.result_widgets_frame,
                                      text="Severity Assessment",
                                      style='ResultTitle.TLabel')
        self.severity_label.grid(row=2, column=0, pady=(0, 5))
        
        self.severity_value = ttk.Label(self.result_widgets_frame,
                                      text="",
                                      style='ResultValue.TLabel')
        self.severity_value.grid(row=3, column=0, pady=(0, 15))
        
        # Additional information
        self.info_label = ttk.Label(self.result_widgets_frame,
                                  text="",
                                  style='ResultText.TLabel',
                                  wraplength=200,
                                  justify='center')
        self.info_label.grid(row=4, column=0, pady=(0, 15))
    
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
        
        # Bind to resize events
        self.canvas.bind('<Configure>', self._on_frame_configure)
        
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
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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
            
            # Load original image
            self.original_image = Image.open(self.image_path)
            
            # Get the frame size
            frame_width = self.canvas.winfo_width()
            frame_height = self.canvas.winfo_height()
            
            # If the canvas hasn't been drawn yet, use a minimum size
            if frame_width <= 1:
                frame_width = 500
            if frame_height <= 1:
                frame_height = 400
            
            # Calculate new size maintaining aspect ratio
            img_aspect = self.original_image.width / self.original_image.height
            frame_aspect = frame_width / frame_height
            
            if img_aspect > frame_aspect:
                # Image is wider than frame
                new_width = frame_width
                new_height = int(frame_width / img_aspect)
            else:
                # Image is taller than frame
                new_height = frame_height
                new_width = int(frame_height * img_aspect)
            
            # Resize image
            resized_image = self.original_image.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS
            )
            self.current_image = ImageTk.PhotoImage(resized_image)
            
            # Center the image in the canvas
            x_center = max(0, (frame_width - new_width) // 2)
            y_center = max(0, (frame_height - new_height) // 2)
            
            # Display image
            self.canvas_image = self.canvas.create_image(
                x_center, y_center,
                anchor=tk.NW,
                image=self.current_image
            )
            
            # Update canvas scrollregion
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
            
            # Hide the initial message and show results
            self.no_results_label.grid_remove()
            self.result_widgets_frame.grid()
            
            # Update probability
            probability_text = f"{int(prediction * 100)}%"
            self.probability_value.configure(text=probability_text)
            
            # Determine severity and color
            if prediction > 0.8:
                severity = "SEVERE"
                color = "#FF0000"  # Red
                info_text = "Immediate medical attention recommended"
            elif prediction > 0.5:
                severity = "MODERATE"
                color = "#FFA500"  # Orange
                info_text = "Medical consultation recommended"
            elif prediction > 0.2:
                severity = "MILD"
                color = "#FFFF52"  # Yellow
                info_text = "Monitor and consult if symptoms worsen"
            else:
                severity = "LOW RISK"
                color = "#008000"  # Green
                info_text = "No immediate action required"
            
            # Update severity display
            self.severity_value.configure(text=severity, foreground=color)
            self.info_label.configure(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
            self.result_widgets_frame.grid_remove()
            self.no_results_label.grid()
            
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
