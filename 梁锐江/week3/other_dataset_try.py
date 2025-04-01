import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import torch.optim as optim

# 1. 定义超参数
# 2. 加载数据
# 3. 创建模型
# 4. 定义损失函数与优化器
# 5. 训练模型

LR = 1e-2
epochs = 10
batch_size = 64


def save_data():
    train_data = FakeData(size=700, transform=ToTensor(), num_classes=5)
    # 初始化列表来保存图像和标签
    all_images = []
    all_labels = []

    # 遍历数据集并收集所有的图像和标签
    for image, label in train_data:
        all_images.append(image)
        all_labels.append(label)
    torch.save({"all_images": all_images, "all_labels": all_labels}, "fake_image_train_data.pt")


model = nn.Sequential(
    nn.Linear(3 * 224 * 224, 256),
    nn.Sigmoid(),
    nn.Linear(256, 5),
)

# 单标签多分类问题损失函数
loss_fn = nn.CrossEntropyLoss
# 优化器
optimizer = optim.SGD(model.parameters(), lr=LR)

if __name__ == '__main__':
    # save_data()
    data = torch.load("fake_image_train_data.pt")
    # torch.Size([3, 224, 224])
    # print(data["all_images"][0].shape)
    l_data = DataLoader(data, batch_size=batch_size, shuffle=True)
    for param, target in l_data:
        print(1)