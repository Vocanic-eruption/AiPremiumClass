import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. 定义神经网络模型（尝试不同结构）
class SimpleNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)  # 输出10类

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DeepNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# 3. 训练与评估函数
def train_and_evaluate(model, learning_rate=0.01, num_epochs=5, output_file="training_results.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with open(output_file, "a") as f:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            log_msg = f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}\n"
            print(log_msg, end='')
            f.write(log_msg)

        # 评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        result_msg = f"Test Accuracy: {accuracy:.2f}%\n"
        print(result_msg)
        f.write(result_msg)
    return accuracy


# 4. 训练并比较不同模型结构的效果
output_file = "training_results.txt"
with open(output_file, "w") as f:
    f.write("Training Results\n================\n")

print("Training SimpleNN with 128 neurons...")
simple_model = SimpleNN(hidden_size=128)
train_and_evaluate(simple_model, learning_rate=0.01, num_epochs=5, output_file=output_file)

print("Training SimpleNN with 128 neurons...")
simple_model = SimpleNN(hidden_size=128)
train_and_evaluate(simple_model, learning_rate=0.001, num_epochs=5, output_file=output_file)

print("Training SimpleNN with 128 neurons...")
simple_model = SimpleNN(hidden_size=128)
train_and_evaluate(simple_model, learning_rate=0.1, num_epochs=5, output_file=output_file)

print("\nTraining SimpleNN with 256 neurons...")
simple_model_256 = SimpleNN(hidden_size=256)
train_and_evaluate(simple_model_256, learning_rate=0.01, num_epochs=5, output_file=output_file)

print("\nTraining DeepNN with 128 neurons...")
deep_model = DeepNN(hidden_size=128)
train_and_evaluate(deep_model, learning_rate=0.01, num_epochs=5, output_file=output_file)
