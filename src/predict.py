import os
import torch
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from models.resnet import ResNetColorizationModel
from models.vgg import VGGColorizationModel
from utils.colorization_utils import ColorizationUtils
from utils.predicting_utils import PredictingUtils


class ModelComparison:
    """Class to handle multi-model colorization comparison."""
    
    def __init__(self, device: str):
        self.device = device
        self.models = {}
        self.configs = {}
        
    def load_model(self, model_name: str, config_path: str, checkpoint_name: str = "best_model.pth") -> bool:
        """Load a model with its configuration."""
        # Load config
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.configs[model_name] = config
        
        # Determine checkpoint path
        model_ckpt_path = os.path.join(config['output']['best_model_dir'], checkpoint_name)
        
        if not os.path.exists(model_ckpt_path):
            print(f"Warning: Checkpoint not found for {model_name} at {model_ckpt_path}")
            return False
        
        # Load model based on type
        if model_name.lower() == "resnet":
            model = ResNetColorizationModel(pretrained=False)
        elif model_name.lower() == "vgg":
            model = VGGColorizationModel(pretrained=False)
        else:
            print(f"Unknown model type: {model_name}")
            return False
        
        # Load weights
        state = torch.load(model_ckpt_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        print(f"{model_name} model loaded successfully from {model_ckpt_path}")
        return True
    
    def predict_single_model(self, model_name: str, input_image: Image.Image) -> Optional[np.ndarray]:
        """Get prediction from a single model."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        # Use appropriate prediction method based on model type
        if model_name.lower() == "vgg":
            # VGG expects PIL image directly
            return PredictingUtils.predict_vgg(model, self.device, input_image)
        else:
            # ResNet expects LLL tensor
            target_size = tuple(self.configs[model_name]["data"]["image_size"])
            lll, _ = ColorizationUtils.preprocess_image(input_image, target_size)
            predicted_ab = PredictingUtils.predict_resnet(model, self.device, lll)
            return ColorizationUtils.reconstruct_image(lll, predicted_ab)
    
    def compare_models(self, image_path: str, output_dir: str, target_size: Tuple[int, int] = (256, 256)):
        """Compare all loaded models on a single image."""
        # Load and preprocess image
        pil_input = Image.open(image_path).convert("RGB")
        resize_transform = transforms.Resize(target_size)
        pil_resized = resize_transform(pil_input)
        
        # Get grayscale
        gray_np = np.array(pil_resized.convert("L"))
        
        # Get ground truth
        gt_np = np.array(pil_resized)
        
        # Get L and ab channels for metrics
        lll, gt_ab_norm = ColorizationUtils.preprocess_image(pil_input, target_size)
        gt_ab_unnorm = gt_ab_norm.detach().cpu().numpy() * ColorizationUtils.AB_SCALE
        
        # Collect predictions from all models
        predictions = {}
        mse_scores = {}
        
        for model_name in self.models:
            pred_rgb = self.predict_single_model(model_name, pil_input)
            if pred_rgb is not None:
                predictions[model_name] = pred_rgb
                
                # Calculate MSE in ab space
                # Convert predicted RGB back to Lab for fair comparison
                pred_pil = Image.fromarray((pred_rgb * 255.0).clip(0, 255).astype("uint8"))
                _, pred_ab_norm = ColorizationUtils.preprocess_image(pred_pil, target_size)
                pred_ab_unnorm = pred_ab_norm.detach().cpu().numpy() * ColorizationUtils.AB_SCALE
                
                mse = np.mean((pred_ab_unnorm - gt_ab_unnorm) ** 2)
                mse_scores[model_name] = mse
        
        # Create comparison plot
        num_models = len(predictions)
        fig, axes = plt.subplots(1, num_models + 2, figsize=(4 * (num_models + 2), 4))
        
        # Grayscale
        axes[0].imshow(gray_np, cmap="gray")
        axes[0].set_title("Grayscale")
        axes[0].axis("off")
        
        # Ground truth
        axes[1].imshow(gt_np)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Model predictions
        for idx, (model_name, pred_rgb) in enumerate(predictions.items()):
            pred_image = (pred_rgb * 255.0).clip(0, 255).astype("uint8")
            axes[idx + 2].imshow(pred_image)
            axes[idx + 2].set_title(f"{model_name}\nMSE: {mse_scores[model_name]:.1f}")
            axes[idx + 2].axis("off")
        
        plt.tight_layout()
        
        # Save plot
        basename = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"comparison_{basename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return mse_scores


def main():
    # Setup device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Initialize comparison tool
    comparator = ModelComparison(device)
    
    # Load models - adjust paths as needed
    models_to_load = [
        ("ResNet", "src/configs/resnet_config.yaml", "best_model.pth"),
        ("VGG", "src/configs/vgg_config.yaml", "best_model_rebal.pth"),  # Use rebalanced weights if available
    ]
    
    loaded_models = []
    for model_name, config_path, checkpoint_name in models_to_load:
        if comparator.load_model(model_name, config_path, checkpoint_name):
            loaded_models.append(model_name)
    
    if len(loaded_models) == 0:
        print("No models could be loaded. Exiting.")
        return
    
    print(f"\nLoaded models: {loaded_models}")
    
    # Get test configuration from first loaded model
    first_config = list(comparator.configs.values())[0]
    test_dir = first_config["testing"]["test_dir"]
    subset_percent = first_config["testing"].get("subset_percent", 1.0)
    target_size = tuple(first_config["data"]["image_size"])
    
    # Create output directory
    output_dir = "outputs/model_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect test images
    img_paths = PredictingUtils.collect_images(test_dir, subset_percent)
    print(f"\nProcessing {len(img_paths)} images...")
    
    # Process each image
    all_mse_scores = {model: [] for model in loaded_models}
    
    for img_path in tqdm(img_paths, desc="Comparing models"):
        mse_scores = comparator.compare_models(img_path, output_dir, target_size)
        
        # Collect MSE scores for summary
        for model_name, mse in mse_scores.items():
            all_mse_scores[model_name].append(mse)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for model_name in loaded_models:
        if model_name in all_mse_scores and len(all_mse_scores[model_name]) > 0:
            scores = all_mse_scores[model_name]
            mean_mse = np.mean(scores)
            std_mse = np.std(scores)
            print(f"{model_name}:")
            print(f"  Mean MSE: {mean_mse:.2f}")
            print(f"  Std MSE:  {std_mse:.2f}")
            print(f"  Min MSE:  {np.min(scores):.2f}")
            print(f"  Max MSE:  {np.max(scores):.2f}")
            print()
    
    # Create summary plot
    if len(all_mse_scores) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Box plot of MSE scores
        data_to_plot = [scores for model, scores in all_mse_scores.items() if len(scores) > 0]
        labels = [model for model, scores in all_mse_scores.items() if len(scores) > 0]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Customize colors
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('MSE Score')
        ax.set_title('Model Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        summary_path = os.path.join(output_dir, "summary_boxplot.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary plot saved to: {summary_path}")
    
    print(f"\nAll comparison images saved to: {output_dir}")


if __name__ == "__main__":
    main()