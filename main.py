import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import re
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

# 检查CUDA是否可用，然后选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设我们的图片库在"./images"目录下
IMAGE_DIR = 'images01'

# 加载预训练的ResNet模型，并将其转移到指定的设备
resnet = models.resnet18(pretrained=True).to(device)

# 假设你的原始图像是灰度图（单通道）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # 将灰度图转换为3通道的伪彩色图
    transforms.Resize((224, 224)), # 调整图像大小
    transforms.ToTensor(), # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 归一化
])

num_ftrs = resnet.fc.in_features
# 移除全连接层，以便我们可以添加我们自己的年龄估计层
modules = list(resnet.children())[:-1]  # 删除最后的全连接层
resnet = torch.nn.Sequential(*modules)


class AgeEstimationModel(torch.nn.Module):
    def __init__(self):
        super(AgeEstimationModel, self).__init__()
        self.resnet = resnet
        # 使用保存的输入特征数量创建新的全连接层
        self.fc = torch.nn.Linear(num_ftrs, 1)  # 预测年龄

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # 将特征展平
        age = self.fc(features)
        return age
# 数据集类
class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        # 从文件名提取年龄，假设格式是 '001A02.jpg' 并且年龄是 '02'
        age_pattern = re.compile(r'\d+A(\d+)')
        match = age_pattern.search(self.image_filenames[idx])
        if match:
            age = int(match.group(1))
        else:
            # 如果没有找到年龄，可以设定一个默认值或者抛出异常
            age = 0  # 或者 raise ValueError("Unable to find age in filename.")

        return image, age

# 创建数据集和数据加载器
dataset = FaceDataset(IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型并将其转移到GPU
model = AgeEstimationModel().to(device)

# 在模型转移到GPU之后初始化优化器
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # 只训练年龄估计层

losses=[]

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, ages in dataloader:
        images, ages = images.to(device), ages.to(device)  # 将数据转移到GPU

        optimizer.zero_grad()
        age_preds = model(images)
        loss = torch.nn.functional.mse_loss(age_preds.squeeze(), ages.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 在每个Epoch结束时计算平均损失并添加到损失列表
    epoch_loss = running_loss / len(dataloader)
    losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

# 保存模型 - 注意：保存的模型可以加载到CPU或其他GPU设备上
torch.save(model.state_dict(), 'age_estimation_model.pth')

# 绘制损失曲线
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs + 1), losses)
plt.title('Training Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
