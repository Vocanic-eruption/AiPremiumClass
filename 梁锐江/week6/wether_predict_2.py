import csv
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.tensorboard import SummaryWriter


class WeatherRnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=1,
            hidden_size=30,
            num_layers=2,
            batch_first=True
        )
        self.fc = torch.nn.Linear(30, 1)

    def forward(self, x):
        out, h_n = self.rnn(x)
        return self.fc(out[:, -1, :])


def load_data():
    with open('./data/Summary of Weather.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        weather_dict = {}
        for line in reader:
            sta = line['STA']
            weather_dict[sta] = weather_dict.get(sta, [])
            weather_dict[sta].append(float(line['MaxTemp']))
        return weather_dict


# 过滤样本量较小的监测站
def data_analysis_distribution(weather_dict):
    temp_lengths = [len(data) for data in weather_dict.values()]
    plt.bar(range(len(temp_lengths)), temp_lengths)
    plt.show()


# 分析异常值
def analytical_outlier(weather_dict):
    # 箱线图数据要针对单一监测站
    # plt.boxplot(weather_dict)
    # plt.show()
    plt.figure(figsize=(20, 250))
    group_count = len(weather_dict) // 2
    values = [value for value in weather_dict.values()]
    for i in range(group_count):
        # 子图1
        data1 = values[i]
        plt.plot(range(len(data1)), data1)
        plt.autoscale()
        plt.subplot(group_count, 2, i * 2 + 1)

        # 子图2
        data2 = values[i * 2 + 1]
        plt.plot(range(len(data2)), data2)
        plt.autoscale()
        plt.subplot(group_count, 2, i * 2 + 2)

    # 调整子图间距
    plt.subplots_adjust(hspace=0.7)
    plt.show()


def data_clean(weather_dict):
    filtered_data = {}
    for sta, value in weather_dict.items():
        std = np.mean(value)
        mean = np.mean(value)

        filtered_data[sta] = [mean if abs(max_temp - std) > 20 else max_temp for max_temp in value]

    return filtered_data


def build_dataset(result_data, total_size, back_steps, predict_steps):
    series = np.zeros((total_size, back_steps + predict_steps))
    sta_idx = np.random.randint(0, len(result_data), total_size)
    total_steps = back_steps + predict_steps

    for i, idx in enumerate(sta_idx):
        value = result_data[idx]
        temp_idx = np.random.randint(0, len(value) - total_steps)
        series[i] = value[temp_idx:temp_idx + total_steps]

    return series[:, :back_steps + 1, np.newaxis].astype(np.float32), series[:, -back_steps, np.newaxis].astype(
        np.float32)


def train_model(train_dl):
    epochs = 10

    writer = SummaryWriter("weather_predict_std")

    model = WeatherRnnModel()
    model.to(torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    global_step = 0
    for epoch in range(epochs):
        for x, y in train_dl:
            x = x.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))
            optimizer.zero_grad()
            predict = model(x)
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()
            global_step += 1
            writer.add_scalar('weather rnn std loss', loss.item(), global_step)

    writer.close()
    return model


def predict_model(model, test_dl):
    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(torch.device('cuda'))
        for x, y in test_dl:
            x = x.to(torch.device('cuda'))
            pred = model(x)
            preds = torch.cat([preds, pred])
    return preds


def plot_series(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$"):
    r, c = 3, 5
    fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))
    for row in range(r):
        for col in range(c):
            plt.sca(axes[row][col])
            ix = col + row * c
            plt.plot(series[ix, :], ".-")
            if y is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :]) + len(y[ix])), y[ix], "bx", markersize=10)
            if y_pred is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :]) + len(y_pred[ix])), y_pred[ix], "ro")
            if y_pred_std is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :]) + len(y_pred[ix])), y_pred[ix] + y_pred_std[ix])
                plt.plot(range(len(series[ix, :]), len(series[ix, :]) + len(y_pred[ix])), y_pred[ix] - y_pred_std[ix])
            plt.grid(True)
            # plt.hlines(0, 0, 100, linewidth=1)
            # plt.axis([0, len(series[ix, :])+len(y[ix]), -1, 1])
            if x_label and row == r - 1:
                plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
                plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.show()


if __name__ == '__main__':
    back_steps = 14
    predict_steps = 1
    total_size = 5000

    # 加载数据
    weather_dict = load_data()

    # 分析并过滤异常值
    # analytical_outlier(weather_dict)
    filtered_data = data_clean(weather_dict)
    # analytical_outlier(filtered_data)

    # 分析并过滤样本量太小的监测站
    # data_analysis_distribution(filtered_data)
    result_data = [value for value in filtered_data.values() if len(value) > 20]

    # 构建数据集
    back_record, predict_record = build_dataset(result_data, total_size, back_steps, predict_steps)
    split_len = int(len(back_record) * 0.8)
    train_data = TensorDataset(torch.from_numpy(back_record[:split_len]), torch.from_numpy(predict_record[:split_len]))
    test_x = back_record[split_len:]
    test_y = predict_record[split_len:]
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_dl = DataLoader(train_data, shuffle=True, batch_size=64)
    test_dl = DataLoader(test_data, shuffle=True, batch_size=64)

    # 模型训练
    model = train_model(train_dl)
    torch.save(model.state_dict(), "weather_predict_std.pth")


    # # 模型预测
    # preds = predict_model(model, test_dl)
    # plot_series(test_x, test_y, preds.cpu().numpy())
