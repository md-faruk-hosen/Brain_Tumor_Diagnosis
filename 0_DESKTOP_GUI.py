import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GradCAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Grad-CAM Visualization")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Define classes
        self.CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize variables
        self.image_path = None
        self.model_path = None
        self.model = None
        self.original_img = None
        self.results = None
        self.current_class_idx = 0

        # Create main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create control frame
        self.create_control_panel()

        # Create visualization frame
        self.create_visualization_panel()

        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Apply style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=3)
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))

        # Initialize transformation for the input image
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_control_panel(self):
        # Control panel frame
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Image selection
        ttk.Label(control_frame, text="Input Image:", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(fill=tk.X, pady=5)

        self.image_path_var = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.image_path_var, width=30).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(image_frame, text="Browse...", command=self.browse_image).pack(side=tk.RIGHT, padx=5)

        # Model selection
        ttk.Label(control_frame, text="Model Path:", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 5))
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)

        self.model_path_var = tk.StringVar()
        self.model_path_var.set("10_Grad_CAM_Results/brain_tumor_model.pth")
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=30).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).pack(side=tk.RIGHT, padx=5)

        # Generate button
        ttk.Button(control_frame, text="Generate Grad-CAM", command=self.generate_gradcam).pack(fill=tk.X, pady=10)

        # Classes radio buttons
        ttk.Label(control_frame, text="Select Class:", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 5))

        self.class_var = tk.IntVar()
        self.class_var.set(0)  # Default to first class

        for i, class_name in enumerate(self.CLASSES):
            ttk.Radiobutton(
                control_frame,
                text=class_name.capitalize(),
                variable=self.class_var,
                value=i,
                command=self.show_selected_class
            ).pack(anchor=tk.W, padx=10)

        # Save results button
        ttk.Button(control_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=(20, 5))

        # Help/info button
        ttk.Button(control_frame, text="Help", command=self.show_help).pack(fill=tk.X, pady=5)

    def create_visualization_panel(self):
        # Visualization panel frame
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding=10)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a frame for the figure
        self.figure_frame = ttk.Frame(self.viz_frame)
        self.figure_frame.pack(fill=tk.BOTH, expand=True)

        # Create a matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add subplots for original image and heatmap
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_title("Original Image")
        self.ax2.set_title("Grad-CAM Visualization")

        self.ax1.axis('off')
        self.ax2.axis('off')

        self.fig.tight_layout()

        # Create a frame for result information
        self.info_frame = ttk.Frame(self.viz_frame)
        self.info_frame.pack(fill=tk.X, pady=10)

        # Prediction information
        self.pred_label = ttk.Label(self.info_frame, text="Prediction: None", font=("Arial", 11))
        self.pred_label.pack(side=tk.LEFT, padx=10)

        # Class probability
        self.prob_label = ttk.Label(self.info_frame, text="Probability: N/A", font=("Arial", 11))
        self.prob_label.pack(side=tk.RIGHT, padx=10)

    def browse_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )
        if filepath:
            self.image_path_var.set(filepath)
            self.image_path = filepath
            self.load_image()

    def browse_model(self):
        filetypes = [
            ("PyTorch models", "*.pth *.pt"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select a model",
            filetypes=filetypes
        )
        if filepath:
            self.model_path_var.set(filepath)
            self.model_path = filepath

    def load_image(self):
        if not self.image_path:
            return

        try:
            # Load and display the original image
            img = cv2.imread(self.image_path)
            if img is None:
                messagebox.showerror("Error", f"Could not read image at {self.image_path}")
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_img = img.copy()

            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()

            # Display original image
            self.ax1.imshow(self.original_img)
            self.ax1.set_title("Original Image")
            self.ax1.axis('off')

            # Clear the heatmap side
            self.ax2.set_title("Grad-CAM Visualization\n(Generate to see results)")
            self.ax2.axis('off')

            self.canvas.draw()

            self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)}")

            # Reset prediction info
            self.pred_label.config(text="Prediction: None")
            self.prob_label.config(text="Probability: N/A")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            self.status_var.set("Error loading image")

    def load_model(self):
        if not self.model_path:
            messagebox.showerror("Error", "Please select a model first")
            return False

        try:
            self.status_var.set("Loading model...")
            self.root.update()

            # Define the model
            class BrainTumorClassifier(torch.nn.Module):
                def __init__(self, num_classes=4):
                    super(BrainTumorClassifier, self).__init__()
                    # Load ResNet50
                    self.model = resnet50(weights=None)

                    # Modify the final fully connected layer
                    num_features = self.model.fc.in_features
                    self.model.fc = torch.nn.Linear(num_features, num_classes)

                    # Save the layer names for Grad-CAM
                    self.gradients = None
                    self.activations = None

                    # Register hook for the last convolutional layer
                    self.model.layer4[-1].conv3.register_forward_hook(self.save_activation)
                    self.model.layer4[-1].conv3.register_full_backward_hook(self.save_gradient)

                def save_activation(self, module, input, output):
                    self.activations = output

                def save_gradient(self, module, grad_input, grad_output):
                    self.gradients = grad_output[0]

                def forward(self, x):
                    return self.model(x)

                def get_activations_gradient(self):
                    return self.gradients

                def get_activations(self):
                    return self.activations

            # Load the model
            self.model = BrainTumorClassifier(num_classes=len(self.CLASSES)).to(self.device)

            try:
                # First try to load with weights_only=True (newer PyTorch versions)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            except TypeError:
                # If weights_only param is not available, use the old way
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            self.model.eval()

            self.status_var.set(f"Model loaded: {os.path.basename(self.model_path)}")
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.status_var.set("Error loading model")
            return False

    def generate_gradcam(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first")
            return

        if not self.model or not os.path.exists(self.model_path_var.get()):
            if not self.load_model():
                return

        try:
            self.status_var.set("Generating Grad-CAM...")
            self.root.update()

            # Get model prediction - first pass without gradients
            with torch.no_grad():
                input_tensor = self.transform(self.original_img).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()

            # Update prediction info
            predicted_class_name = self.CLASSES[predicted_class].capitalize()
            self.pred_label.config(text=f"Prediction: {predicted_class_name}")

            # Generate Grad-CAM for all classes
            self.results = []

            # Need a new forward pass with gradients enabled for Grad-CAM
            input_tensor = self.transform(self.original_img).unsqueeze(0).to(self.device)
            input_tensor.requires_grad = True

            for class_idx, class_name in enumerate(self.CLASSES):
                # Forward pass with gradients
                self.model.zero_grad()
                outputs = self.model(input_tensor)

                # Target for backprop - one-hot encoding for the current class
                target = torch.zeros(outputs.size()).to(self.device)
                target[0, class_idx] = 1

                # Backward pass
                outputs.backward(gradient=target, retain_graph=True)

                # Get gradients and activations
                gradients = self.model.get_activations_gradient()
                activations = self.model.get_activations()

                if gradients is None or activations is None:
                    self.status_var.set(f"Warning: Gradients or activations are None for class {class_name}")
                    continue

                # Global average pooling
                weights = torch.mean(gradients, dim=[2, 3])[0, :]

                # Create class activation map
                cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(self.device)

                # Weight the channels by corresponding gradients
                for i, w in enumerate(weights):
                    cam += w * activations[0, i, :, :]

                # Apply ReLU
                cam = F.relu(cam)

                # Normalize
                cam = cam - torch.min(cam)
                if torch.max(cam) > 0:
                    cam = cam / torch.max(cam)

                # Convert to numpy and resize to match original image
                cam_np = cam.cpu().detach().numpy()
                cam_np = cv2.resize(cam_np, (self.original_img.shape[1], self.original_img.shape[0]))

                # Convert CAM to heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                # Combine original image with heatmap
                img_float = self.original_img.astype(float) / 255
                heatmap_float = heatmap.astype(float) / 255
                overlay = 0.7 * img_float + 0.3 * heatmap_float
                overlay = np.clip(overlay, 0, 1)

                # Store the results
                self.results.append((heatmap, overlay, class_name, probabilities[class_idx].item()))

            # Display results for the selected class
            self.show_selected_class()

            self.status_var.set("Grad-CAM generation completed!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error generating Grad-CAM: {str(e)}")
            self.status_var.set("Error generating Grad-CAM")

    def show_selected_class(self):
        if not self.results:
            return

        class_idx = self.class_var.get()
        if class_idx < 0 or class_idx >= len(self.results):
            return

        # Get the results for the selected class
        _, overlay, class_name, probability = self.results[class_idx]

        # Update the visualization
        self.ax1.clear()
        self.ax2.clear()

        # Display original image
        self.ax1.imshow(self.original_img)
        self.ax1.set_title("Original Image")
        self.ax1.axis('off')

        # Display Grad-CAM visualization
        self.ax2.imshow(overlay)
        self.ax2.set_title(f"Grad-CAM: {class_name.capitalize()}")
        self.ax2.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

        # Update probability label
        self.prob_label.config(text=f"Probability: {probability:.4f}")

    def save_results(self):
        if not self.results:
            messagebox.showerror("Error", "No results to save. Please generate Grad-CAM first.")
            return

        # Ask for directory to save results
        output_dir = filedialog.askdirectory(title="Select directory to save results")
        if not output_dir:
            return

        try:
            os.makedirs(output_dir, exist_ok=True)

            # Create a combined figure showing all classes
            plt.figure(figsize=(15, 10))

            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(self.original_img)
            plt.title(f"Original Image")
            plt.axis('off')

            # Grad-CAM for each class
            for i, (_, overlay, class_name, probability) in enumerate(self.results):
                plt.subplot(2, 3, i + 2)
                plt.imshow(overlay)
                plt.title(f"Grad-CAM: {class_name.capitalize()}\nProbability: {probability:.4f}")
                plt.axis('off')

            plt.tight_layout()

            # Save combined figure
            img_basename = os.path.basename(self.image_path)
            output_path = os.path.join(output_dir, f"gradcam_{img_basename}_all_classes.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save individual class visualizations
            for _, overlay, class_name, probability in self.results:
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(self.original_img)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(overlay)
                plt.title(f"Grad-CAM: {class_name.capitalize()}\nProbability: {probability:.4f}")
                plt.axis('off')

                plt.tight_layout()

                # Save figure
                output_path = os.path.join(output_dir, f"gradcam_{img_basename}_{class_name}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

            messagebox.showinfo("Success", f"Results saved to {output_dir}")
            self.status_var.set(f"Results saved to {output_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving results: {str(e)}")
            self.status_var.set("Error saving results")

    def show_help(self):
        help_text = """
        Grad-CAM Brain Tumor Visualization Tool

        This tool visualizes which regions of brain tumor images the model is focusing on when making predictions.

        Usage:
        1. Select an input image using the 'Browse' button
        2. Specify the path to a trained model (default is provided)
        3. Click 'Generate Grad-CAM' to create visualizations
        4. Use the radio buttons to view different tumor classes
        5. Click 'Save Results' to save all visualizations

        The heat map shows which areas of the image are contributing most to the classification.
        Red/yellow areas indicate regions the model focuses on for a particular class.

        Classes:
        - Glioma: A type of tumor that occurs in the brain and spinal cord
        - Meningioma: A tumor that forms on membranes covering the brain and spinal cord
        - No Tumor: Normal brain tissue without tumors
        - Pituitary: A tumor that forms in the pituitary gland
        """

        messagebox.showinfo("Help", help_text)


def main():
    root = tk.Tk()
    app = GradCAMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()