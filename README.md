# 代码说明

CNN有额外数据增强：python cnn_cifar.py

CNN无额外数据增强：python cnn_cifar_woa.py

Transformer有额外数据增强:python transformer_cifar.py

Transformer无额外数据增强:python transformer_cifar_2.py

在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型


# TensorFlow CIFAR-100 图像分类

这个脚本使用TensorFlow 2实现了一个用于CIFAR-100图像分类任务的卷积神经网络(CNN)。CIFAR-100数据集是一个包含100个类别，每个类别包含600张32x32彩色图像的数据集。

## 特性

- **多层卷积单元**：网络由多个卷积单元组成，每个单元包含两个卷积层和一个最大池化层。
- **全连接层**：在卷积层之后，使用全连接层进行分类。
- **预处理**：将输入图像归一化到[0,1]区间，并进行one-hot编码。
- **训练与评估**：在训练过程中计算损失，并在测试集上评估准确率。

## 环境要求

- Python 3.x
- TensorFlow 2.x

可以通过以下命令安装所需库：
```bash
pip install -r requirements.txt
```
## 使用方法

1. **准备数据集**：脚本会自动下载CIFAR-100数据集。
2. **运行脚本**：使用Python运行脚本。脚本将训练CNN模型，并在测试集上计算准确率。
3. **训练过程**：脚本将进行50个epoch的训练，并在每个epoch结束后打印损失和准确率。

## 代码结构

- `conv_layers`：定义了网络的卷积层。
- `preprocess`：定义了数据的预处理函数，包括归一化和one-hot编码。
- `main`：主函数，用于构建模型、训练和评估。

## 模型细节

- 模型使用Sequential API快速构建。
- 卷积层使用ReLU激活函数，最大池化层用于下采样。
- 全连接层使用ReLU激活函数，最后一个全连接层输出100维，对应100个类别。

## 训练过程

- 使用Adam优化器，学习率为1e-4。
- 在每个batch中计算梯度，并更新模型权重。
- 每100个step打印一次损失。

## 评估

- 在每个epoch结束后，在测试集上计算模型的准确率。

## 注意事项

- 确保TensorFlow 2.x正确安装。
- 根据实际环境调整学习率和其他超参数。


# 使用Vision Transformer (ViT) 对 CIFAR-100 数据集进行分类

使用PyTorch和预训练的Vision Transformer模型对CIFAR-100数据集进行图像分类。CIFAR-100是一个包含100个类别的图像数据集，每个类别包含600张32x32的彩色图像。

## 特性

- **预训练模型**：使用预训练的Vision Transformer模型进行迁移学习。
- **数据增强**：包括随机裁剪、随机水平翻转和标准化。
- **CutMix**：一种数据增强技术，通过在图像间进行区域混合来提高模型的泛化能力。
- **学习率调度**：使用余弦退火学习率调度器调整训练过程中的学习率。
- **早停法**：当验证集损失在一定epoch内不再下降时停止训练以避免过拟合。
- **TensorBoard**：可视化训练和验证过程中的损失和准确率。

## 环境要求

- Python 3.x
- PyTorch
- torchvision
- timm
- numpy
- tqdm
- torch.optim.lr_scheduler

可以通过以下命令安装所需库：

```bash
pip install torch torchvision timm numpy tqdm
```

## 使用方法

1. **准备数据集**：脚本会自动下载CIFAR-100数据集并进行预处理。
2. **配置模型和训练参数**：在脚本顶部的配置部分，根据需要调整模型和训练参数。
3. **运行脚本**：使用Python运行脚本。脚本将训练ViT模型，并在测试集上计算准确率。
4. **训练过程**：脚本将进行指定的epoch数的训练，并在每个epoch结束后打印当前的损失和准确率。

## 代码结构

- **配置部分**：定义了设备、批量大小、epoch数、学习率等配置。
- **数据加载**：使用`torchvision.datasets.CIFAR100`加载数据，并使用`DataLoader`进行批处理和多线程加载。
- **模型定义**：使用`timm`库创建Vision Transformer模型，并调整分类器以适应100个类别。
- **训练和验证函数**：定义了训练和验证过程中的逻辑，包括CutMix数据增强和损失计算。
- **主训练循环**：实现了训练、验证和保存最佳模型的逻辑。

## 模型细节

- 使用`vit_tiny_patch16_224`作为基础模型，并将其分类器头部替换为100维输出以匹配CIFAR-100的类别数。

## 训练和验证

- 使用交叉熵损失函数和AdamW优化器进行训练。
- 在每个epoch结束后，在验证集上评估模型性能，并使用TensorBoard记录训练和验证的损失。

## 测试

- 加载保存的最佳模型，并在测试集上计算整体准确率。

## 注意事项

- 确保PyTorch和相关库正确安装。
- 根据实际环境调整学习率、批量大小和其他超参数。

