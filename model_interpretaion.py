import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from captum.attr import IntegratedGradients, visualization

def interpret_prediction(model, image_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_class = output.argmax(dim=1).item()
    predicted_label = class_names[predicted_class]
    
    # Initialize IntegratedGradients
    ig = IntegratedGradients(model)
    
    # Compute attributions
    attributions = ig.attribute(input_tensor, target=predicted_class, n_steps=200)
    
    # Visualize attributions
    original_image = np.array(image)
    attr_image = visualization.visualize_image_attr(
        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(original_image, (1, 2, 0)),
        method="heat_map",
        sign="positive",
        show_colorbar=True,
        title=f"Prediction: {predicted_label}"
    )
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(attr_image[0])
    plt.title("Attribution Map")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("interpretation_result.png")
    plt.close()
    
    return predicted_label, "interpretation_result.png"

# Function to get top-k predictions
def get_top_k_predictions(model, image_path, class_names, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get top-k predictions
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_k_probs, top_k_indices = torch.topk(probabilities, k)
    
    results = []
    for i in range(k):
        class_idx = top_k_indices[0][i].item()
        prob = top_k_probs[0][i].item()
        class_name = class_names[class_idx]
        results.append((class_name, prob))
    
    return results

# Usage example
if __name__ == "__main__":
    # Assume you have a trained model and class_names list
    model = CustomViT(num_classes=len(class_names))
    model.load_state_dict(torch.load("custom_vit_trained_model.pth"))
    
    image_path = ""
    
    # Interpret the prediction
    predicted_label, interpretation_image = interpret_prediction(model, image_path, class_names)
    print(f"Predicted label: {predicted_label}")
    print(f"Interpretation image saved as: {interpretation_image}")
    
    # Get top-5 predictions
    top_5_predictions = get_top_k_predictions(model, image_path, class_names, k=5)
    print("\nTop 5 predictions:")
    for label, prob in top_5_predictions:
        print(f"{label}: {prob:.4f}")