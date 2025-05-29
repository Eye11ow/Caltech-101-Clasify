import argparse
from data import load_datasets
from model import create_model
from train import train_model
from predict import predict_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train or Predict with Caltech-101 dataset.')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction')
    parser.add_argument('--model_path', type=str, help='Path to the trained model for prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate ')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001, help='Fine tuning learning rate for pre-trained model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device to use: cpu or cuda')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if device is cuda (e.g., 0, 1, 2, 3)')
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES environment variable if using GPU
    if args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        train_loader, val_loader = load_datasets(batch_size=args.batch_size)
        
        # Create TensorBoard writer
        writer = SummaryWriter(f'runs/caltech101_experiment_{datetime.now().strftime("%Y%m%d_%H:%M:%S")}')

        # Train with pre-trained weights
        print("Training with pre-trained weights...")
        model_pretrained = create_model(pretrained=True).to(device)
        writer.add_graph(model_pretrained, next(iter(train_loader))[0].to(device))

        # Freeze all layers except the final layer
        for param in model_pretrained.parameters():
            param.requires_grad = False
        for param in model_pretrained.fc.parameters():
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {'params': model_pretrained.fc.parameters(), 'lr': args.learning_rate},
            {'params': [p for n, p in model_pretrained.named_parameters() if p.requires_grad and not n.startswith('fc')], 'lr': args.fine_tune_lr}
        ])

        train_model(model_pretrained, train_loader, val_loader, optimizer, criterion, 
                    args.num_epochs, writer, device, "pretrained")

        # Train from scratch
        print("\nTraining from scratch...")
        model_scratch = create_model(pretrained=False).to(device)
        optimizer = optim.Adam(model_scratch.parameters(), lr=args.learning_rate)

        train_model(model_scratch, train_loader, val_loader, optimizer, criterion, 
                    args.num_epochs, writer, device, "scratch")

        writer.close()

    elif args.mode == 'predict':
        if not args.image_path or not args.model_path:
            raise ValueError("--image_path and --model_path are required for prediction mode.")
        
        class_index = predict_image(args.image_path, args.model_path, device=device)
        print(f"The predicted class index is: {class_index}")

if __name__ == "__main__":
    main()
