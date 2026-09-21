import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import logging

from config import Config

logger = logging.getLogger("pest_inference")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler("inference.log", mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class PestInference:
    def __init__(self, model_path=None, num_classes=None, device=None, class_names=None):
        self.device = torch.device(Config.DEVICE) if device is None else device
        self.model_path = model_path or Config.INFERENCE_MODEL
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self._load_model()

    def _load_model(self):
        logger.info("Loading trained model for inference...")
        try:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully!")
            return model
        except FileNotFoundError:
            logger.error(f"Model not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, image_path):
        if not os.path.exists(image_path):
            logger.error(f"Sample image not found at {image_path}")
            raise FileNotFoundError(f"Sample image not found at {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            raise

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = probabilities.topk(1, dim=1)
            top_prob = top_prob.cpu().numpy()[0][0]
            top_idx = top_idx.cpu().numpy()[0][0]
            class_name = self.class_names[top_idx] if self.class_names else f"Class {top_idx}"

            # Classwise probabilities
            classwise_probs = probabilities.cpu().numpy()[0]
            classwise_dict = {
                self.class_names[i]: float(classwise_probs[i])
                for i in range(self.num_classes)
            }

            result = {
                "image_path": image_path,
                "predicted_class": class_name,
                "predicted_index": int(top_idx),
                "confidence": float(top_prob),
                "classwise_probabilities": classwise_dict
            }
            logger.info(f"Prediction for {image_path}: {class_name} (index {top_idx}), Confidence: {top_prob:.4f}")
            return result

    def predict_to_json(self, image_path, json_path=None):
        result = self.predict(image_path)
        json_str = json.dumps(result, indent=2)
        if json_path:
            with open(json_path, "w") as f:
                f.write(json_str)
            logger.info(f"Prediction result saved to {json_path}")
        return json_str

def inferencer(image_paths, model_path=None, num_classes=None, device=None, class_names=None):
    """
    Perform inference on a list of image paths.

    Args:
        image_paths (list): List of image file paths.
        model_path (str, optional): Path to the trained model.
        num_classes (int, optional): Number of classes.
        device (torch.device or str, optional): Device to use.
        class_names (list, optional): List of class names.

    Returns:
        list: List of inference results (dicts) for each image.
    """
    pest_infer = PestInference(
        model_path=model_path,
        num_classes=num_classes,
        device=device,
        class_names=class_names
    )
    results = []
    for img_path in image_paths:
        try:
            result = pest_infer.predict(img_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Inference failed for {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "error": str(e)
            })
    return results
