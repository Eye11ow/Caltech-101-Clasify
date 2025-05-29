import torch
from PIL import Image
from torchvision import transforms
from model import create_model
import matplotlib.pyplot as plt

def predict_image(image_path, model_path, device=torch.device('cpu'), num_classes=101):
    model = create_model(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    
    # Get the predicted class index
    predicted_class_index = predicted.item()
    
    # Visualize the image and prediction result
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class_index}')
    plt.axis('off')
    plt.show()
    
    return predicted_class_index



