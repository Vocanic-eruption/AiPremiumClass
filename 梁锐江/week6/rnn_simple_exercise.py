import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=28,
            hidden_size=50,
            bias=True,
            num_layers=5,
            batch_first=True
        )
        self.fc = torch.nn.Linear(50, 10)

    def forward(self, x):
        outputs, l_h = self.rnn(x)
        out = self.fc(outputs[:, -1, :])
        return out


def load_model_param():
    pass


if __name__ == '__main__':

    # writer = SummaryWriter()
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    train_data = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    #
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=64, shuffle=True)
    #
    # model = RNNClassifier()
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_func = torch.nn.CrossEntropyLoss()
    #
    # num_epochs = 10
    # global_step = 0  # 初始化全局步数
    # for epoch in range(num_epochs):
    #     model.train()
    #     for i, (data, target) in enumerate(train_dl):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data.squeeze())
    #         loss = loss_func(output, target)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()
    #         if i % 100 == 0:
    #             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    #             writer.add_scalar('loss', loss.item(), global_step)
    #         global_step += 1
    #
    #     model.eval()
    #     with torch.no_grad():
    #         total = 0
    #         acc = 0
    #         for test_data, test_target in test_dl:
    #             test_data, test_target = test_data.to(device), test_target.to(device)
    #             # 评估模式
    #             test_out = model(test_data.squeeze())
    #             values, predict = torch.max(test_out, 1)
    #
    #             total += test_target.size(0)
    #             acc += (predict == test_target).sum().item()
    #         print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy:{acc / total * 100}%")
    #         writer.add_scalar('test accuracy', acc, global_step)
    #
    # writer.close()

    # 通过对象直接存储
    # torch.save(model, 'rnn_model.pth')
    model = torch.load('rnn_model.pth', weights_only=False)

    # 使用模型进行预测
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)
            predict = model(data.squeeze())
            _, pred_clazz = torch.max(predict, 1)
            print((pred_clazz == target).sum().item())
            break

    # # 只保存参数权重
    # torch.save(model.state_dict(), 'rnn_model.pth')
    # model = RNNClassifier()
    # torch.load_state_dict(torch.load('rnn_model.pth'))
