# Caltech-101 图像分类项目

本项目展示了如何使用 Caltech-101 数据集进行图像分类任务。项目使用预训练的 ResNet-18 模型进行迁移学习，并从头开始训练一个模型。

## 目录
1. [概述](#概述)
2. [依赖项](#依赖项)
3. [数据准备](#数据准备)
4. [训练](#训练)
5. [预测](#预测)
6. [结果可视化](#结果可视化)
7. [致谢](#致谢)

## 概述
Caltech-101 数据集包含 9,146 张图像，分为 101 类别。该项目包括以下脚本：
- 使用预训练的 ResNet-18 模型在 Caltech-101 数据集上进行训练。
- 微调预训练模型的最后一层。
- 从头开始训练 ResNet-18 模型。
- 使用训练好的模型对新图像进行类别预测。
- 使用 TensorBoard 可视化训练结果。

## 依赖项
确保安装了以下依赖项：





```
bash
pip install torch torchvision matplotlib tensorboard
```

## 数据准备
1. 从官方源或其他可靠仓库下载 Caltech-101 数据集。
2. 将下载的 tar.gz 文件（`101_ObjectCategories.tar.gz`）放在 `caltech-101` 目录中。
3. 脚本会自动解压数据集（如果尚未解压）。

## 训练
要训练模型，请使用 `main.py` 脚本。您可以指定各种超参数和设备设置。

### 示例命令

#### 使用预训练权重训练
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --batch_size 64 --num_epochs 50 --pre_learning_rate 0.0005 --fine_tune_lr 0.00005 --sc_learning_rate 0.0005 --device cuda --gpu_id 0
```

#### 从头开始训练
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --batch_size 64 --num_epochs 50 --pre_learning_rate 0.0005 --fine_tune_lr 0.00005 --sc_learning_rate 0.0005 --device cuda --gpu_id 0
```

### 命令行参数
- `--mode`: 操作模式 (`train` 或 `predict`)。
- `--image_path`: 预测模式下用于预测的图像路径（必需）。
- `--model_path`: 预测模式下使用的训练好的模型路径（必需）。
- `--batch_size`: 训练和验证的批量大小。
- `--num_epochs`: 训练轮数。
- `--pre_learning_rate`: 预训练模型微调的学习率。
- `--fine_tune_lr`: 预训练模型微调的学习率。
- `--sc_learning_rate`: 从头开始训练模型的学习率。
- `--device`: 使用的设备 (`cpu` 或 `cuda`)。
- `--gpu_id`: 如果设备是 `cuda`，则指定 GPU 编号。

## 预测
要使用训练好的模型预测图像的类别标签，请使用 `main.py` 脚本中的 `predict` 模式。

### 示例命令
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode predict --image_path test_image.jpg --model_path best_pretrained_model_20250529_132430.pth --device cuda --gpu_id 0
```

## 结果可视化
在训练过程中，TensorBoard 日志会被生成以可视化训练和验证损失及准确率。要查看这些日志，请运行：

```bash
tensorboard --logdir runs/
```

打开提供的 URL 在浏览器中查看可视化结果。

## 致谢
- Caltech-101 数据集在其各自的许可证下使用。
- 使用 PyTorch 和 torchvision 库进行深度学习任务。
- 使用 Matplotlib 进行可视化。
- 使用 TensorBoard 监控训练进度。


