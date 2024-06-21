import os
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4  # 正则项


# # Data augmentation and normalization for training
# transform_train = transforms.Compose([
#     transforms.Resize(224),  # Resize the image to 224x224
#     transforms.RandomCrop(224, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
# ])

transform_train = transforms.Compose([
    transforms.Resize(224), 
    # transforms.RandomCrop(224, padding=4),
    # transforms.RandomHorizontalFlip(), 
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20), # 随机旋转图像，最大旋转角度为20度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变亮度、对比度、饱和度和色调
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])


transform_test = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.CenterCrop(224), # 中心剪裁
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(ViTClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)

    
    
def train(epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 计算loss
        train_loss += loss.item()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss=train_loss / (i + 1))
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader)
    train_accuracy = correct / total
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
            # 计算loss
            val_loss += loss.item()
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(val_loss=val_loss / (i + 1))
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy



# Load datasets
train_val_set = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_set = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Split training and validation sets
num_train = len(train_val_set)
split = int(np.floor(0.1 * num_train))
train_set, val_set = random_split(train_val_set, [num_train - split, split], generator=torch.Generator().manual_seed(42))

# Data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
model = ViTClassifier(num_classes=100).to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)


def main():
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join('trans_woa', 'CIFAR100_ViT_experiment'))
    
    # Early stopping
    early_stopping_patience = 10
    best_val_loss = float('inf')
    counter = 0
    
    # Training
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
            torch.save(model.state_dict(), 'trans_woa/best_model_woa.pth')
        else:
            counter += 1
        if counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Testing
    model.load_state_dict(torch.load('trans_woa/best_model_woa.pth'))
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
