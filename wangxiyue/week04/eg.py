import torch
from sklearn.datasets import fetch_olivetti_faces


x,y=  fetch_olivetti_faces(data_home='../data/face_data', return_X_y=True)
mode_file_path = '/tmp/pycharm_project_535/wangxiyue/data/model/torch_nn_module_258.pth'
model = torch.load(mode_file_path,map_location=torch.device('cuda'))
model.eval()

with (torch.no_grad()):

    acc = 0
    test_data_size = 0
    x= torch.tensor(x).to('cuda')
    y= torch.tensor(y).to('cuda')

    # # for i in range(len(y)) : # 80
    #     data = x[i].to('cuda')
    #     labels = y[i].to('cuda')
    data = x.to('cuda')
    labels = y.to('cuda')
    pred_test = model(data)
    acc += (pred_test.argmax(1) == labels).sum().item()
    test_data_size += labels.size(0)
    print(
        f'accSize={acc},test_data_size={test_data_size} , Acc :{(acc / test_data_size * 100):.3f}%')
