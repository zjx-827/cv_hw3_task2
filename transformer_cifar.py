import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.optim as optim
from torchvision.datasets import CIFAR100
import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.transforms import functional as F

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
learning_rate = 1e-3
alpha = 1.0
cutmix_prob = 0.5


# # Data augmentation and normalization for training
# transform_train = transforms.Compose([
#     transforms.Resize(224),  # Resize the image to 224x224
#     transforms.RandomCrop(224, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(20), # 随机旋转图像，最大旋转角度为20度
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变亮度、对比度、饱和度和色调
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
# ])

# # Normalization for validation and testing
# transform_test = transforms.Compose([
#     transforms.Resize(224),  # Resize the image to 224x224
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
# ])



transform_train = transforms.Compose([
    transforms.Resize(224), 
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(20), # 随机旋转图像，最大旋转角度为20度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变亮度、对比度、饱和度和色调
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])


transform_test = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.CenterCrop(224), # 中心剪裁
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])



# Load datasets
train_val_set = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_set = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Split training and validation
num_train = len(train_val_set)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))

train_set, val_set = random_split(train_val_set, [num_train - split, split])

# Data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def cutmix_data(inputs, labels, alpha):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size()[0]).to(device)

    target_a = labels
    target_b = labels[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    return inputs, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# Define the model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(ViTClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)

model = ViTClassifier(num_classes=100).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Early stopping
early_stopping_patience = 10
best_val_loss = float('inf')
counter = 0

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if np.random.rand() < cutmix_prob:
            inputs, target_a, target_b, lam = cutmix_data(inputs, labels, alpha)
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss=running_loss / (i + 1))
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy

    

def validate(epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    with torch.no_grad():
        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loop.set_postfix(val_loss=val_loss / (i + 1))
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    return val_loss, val_accuracy
     

def main():
    writer = SummaryWriter('trans_aug/CIFAR100_ViT_experiment')
    # Main training loop
    for epoch in range(epochs):
        train_loss, train_accuracy = train(epoch)
        val_loss, val_accuracy = validate(epoch)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'trans_aug/best_model_aug.pth')
        else:
            counter += 1
        if counter >= early_stopping_patience:
            print("Early stopping triggered")
            break


    # Testing
    model.load_state_dict(torch.load('trans_aug/best_model_aug.pth'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

    writer.close()
    
if __name__ == "__main__":
    main()
