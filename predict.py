"""
Prediction Script for Skin Disease Classification
Load trained models and predict on new images
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

# Disease information
DISEASE_INFO = {
    "BA-cellulitis": {
        "type": "Bacterial",
        "description": "A bacterial skin infection that causes redness, swelling, and pain."
    },
    "BA-impetigo": {
        "type": "Bacterial",
        "description": "A highly contagious bacterial skin infection forming sores and blisters."
    },
    "FU-athlete-foot": {
        "type": "Fungal",
        "description": "A fungal infection affecting the feet, causing itching and peeling."
    },
    "FU-nail-fungus": {
        "type": "Fungal",
        "description": "A fungal infection of the nails causing discoloration and thickening."
    },
    "FU-ringworm": {
        "type": "Fungal",
        "description": "A fungal infection causing circular, ring-shaped rashes on the skin."
    },
    "PA-cutaneous-larva-migrans": {
        "type": "Parasitic",
        "description": "A parasitic skin infection caused by hookworm larvae."
    },
    "VI-chickenpox": {
        "type": "Viral",
        "description": "A viral infection causing itchy rash with small, fluid-filled blisters."
    },
    "VI-shingles": {
        "type": "Viral",
        "description": "A viral infection causing painful rash, caused by reactivation of chickenpox virus."
    }
}


def create_googlenet_model(num_classes):
    """Create GoogLeNet model architecture matching training."""
    model = models.googlenet(weights=None, aux_logits=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def create_mobilenet_model(num_classes):
    """Create MobileNet V2 model architecture."""
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def load_model(model_path, model_type='googlenet'):
    """Load a trained model from checkpoint."""
    if model_type.lower() == 'googlenet':
        model = create_googlenet_model(len(CLASS_NAMES))
    elif model_type.lower() == 'mobilenet':
        model = create_mobilenet_model(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model with best accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model


def preprocess_image(image_path):
    """Preprocess an image for prediction."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def predict(model, image_tensor):
    """Make prediction on a single image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def predict_with_ensemble(image_path, googlenet_model, mobilenet_model):
    """Make prediction using ensemble of both models."""
    image_tensor, original_image = preprocess_image(image_path)
    
    # Get predictions from both models
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        googlenet_outputs = googlenet_model(image_tensor)
        mobilenet_outputs = mobilenet_model(image_tensor)
        
        # Average the probabilities
        googlenet_probs = torch.softmax(googlenet_outputs, dim=1)
        mobilenet_probs = torch.softmax(mobilenet_outputs, dim=1)
        
        ensemble_probs = (googlenet_probs + mobilenet_probs) / 2
        confidence, predicted = torch.max(ensemble_probs, 1)
    
    return predicted.item(), confidence.item(), ensemble_probs[0].cpu().numpy(), original_image


def display_prediction(image, predicted_class, confidence, probabilities, title="Prediction"):
    """Display the image with prediction results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    disease_name = CLASS_NAMES[predicted_class]
    disease_type = DISEASE_INFO[disease_name]['type']
    ax1.set_title(f"Predicted: {disease_name}\nType: {disease_type}\nConfidence: {confidence*100:.2f}%", 
                  fontsize=12)
    
    # Show probability bar chart
    colors = ['green' if i == predicted_class else 'steelblue' for i in range(len(CLASS_NAMES))]
    bars = ax2.barh(CLASS_NAMES, probabilities * 100, color=colors)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print disease information
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Disease: {disease_name}")
    print(f"Type: {disease_type}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Description: {DISEASE_INFO[disease_name]['description']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Skin Disease Classification Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='ensemble', 
                       choices=['googlenet', 'mobilenet', 'ensemble'],
                       help='Model to use for prediction')
    parser.add_argument('--googlenet_path', type=str, default='GoogLeNet_best.pth',
                       help='Path to GoogLeNet model checkpoint')
    parser.add_argument('--mobilenet_path', type=str, default='MobileNetV2_best.pth',
                       help='Path to MobileNet model checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"Using device: {device}")
    print(f"Processing image: {args.image}")
    
    if args.model == 'ensemble':
        # Load both models for ensemble prediction
        print("\nLoading models for ensemble prediction...")
        googlenet_model = load_model(args.googlenet_path, 'googlenet')
        mobilenet_model = load_model(args.mobilenet_path, 'mobilenet')
        
        predicted_class, confidence, probabilities, image = predict_with_ensemble(
            args.image, googlenet_model, mobilenet_model
        )
    else:
        # Use single model
        model_path = args.googlenet_path if args.model == 'googlenet' else args.mobilenet_path
        print(f"\nLoading {args.model} model...")
        model = load_model(model_path, args.model)
        
        image_tensor, image = preprocess_image(args.image)
        predicted_class, confidence, probabilities = predict(model, image_tensor)
    
    display_prediction(image, predicted_class, confidence, probabilities)


if __name__ == "__main__":
    main()
