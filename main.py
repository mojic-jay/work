import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转（角度范围：-10到10度）
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(root='train', transform=transform)
test_dataset = datasets.ImageFolder(root='test', transform=transform)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 显示部分训练图像，确保数据读取正确
import numpy as np


def show_images(data_loader, rows=4, cols=4):
    """
    从 DataLoader 中显示一批图像
    """
    # 从 DataLoader 中获取一个批次的数据
    images, labels = next(iter(data_loader))

    # 转换到 NumPy 格式（如果是张量）
    # 定义反归一化函数
    def unnormalize(img, mean, std):
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)  # 逆标准化
        return img

    # 反归一化图像
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = torch.stack([unnormalize(img.clone(), mean, std) for img in images])

    # 转换为 NumPy 格式
    images = images.numpy().transpose(0, 2, 3, 1)  # 转换为 (batch_size, height, width, channels)
    images = np.clip(images, 0, 1)  # 确保在 [0, 1] 范围内

    plt.figure(figsize=(12, 8))  # 调整图片总布局大小
    for i in range(min(len(images), rows * cols)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')   # 假设是灰度图像
        plt.title(f"Label: {labels[i].item()}", fontsize=8)  # 显示标签
        plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


show_images(train_loader)

# 使用ResNet18作为基础模型
model = models.resnet18(pretrained=True)

# 修改ResNet18的全连接层
model.fc = nn.Linear(model.fc.in_features, 4)  # 输出4个类别

# 打印模型结构
print(model)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 适合多分类问题的交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # 切换为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  # 更新学习率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return train_losses, train_accuracies


# 训练模型
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)

# 绘制损失函数与准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# 模型评估
def evaluate_model(model, test_loader):
    model.eval()  # 切换为评估模式
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.numpy())
            predictions.extend(predicted.numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')


# 评估模型
evaluate_model(model, test_loader)
