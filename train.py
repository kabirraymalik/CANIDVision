import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import FrameDataset
from models import VideoClassifier
import utils
    
def train():

    #device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #hyperparams
    num_classes = 101
    sequence_length = 20
    frames_per_vid = 10
    input_size = 2048 #resnet-50 output going into lstm 
    num_layers = 2
    hidden_size = 256
    num_epochs = 200
    batch_size = 10
    learning_rate = 0.002
    train_split = 0.8

    #readability stuff
    if torch.cuda.is_available() == False:
        print("cuda unavailable, running on cpu")
    accuracies = []
    losses = []

    #define transform
    transform = transforms.Compose([
        transforms.Resize((229,229)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ])

    # Initialize dataset
    dataset = FrameDataset(frames_per_vid, transform=transform)

    # Splitting dataset into train and test
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #defining model
    model = VideoClassifier(input_size, hidden_size, num_layers, num_classes)
    model.to(device)

    #loss and optimizer
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #initialize utils for evaluation
    learning_utils = utils.learningUtils()

    #training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # original shape: [100, 1, 28, 28]
            # resized shape: [100, 28, 28] 
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = criteria(outputs, labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #ouputs
            if (i+1) % 100 == 0:
                print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')
                losses.append(loss.item())
    #test
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                #this is value, index, but only care about index
                _, predictions = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()

            accuracy = 100 * n_correct / n_samples
            accuracies.append(accuracy)
            print(f"accuracy: {accuracy:.3f}")
            learning_utils._save_stats(accuracies, losses)
            learning_utils._save_model(model)

if __name__ == "__main__":
    train()
