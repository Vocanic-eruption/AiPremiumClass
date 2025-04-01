import torch

# 检查pytorch是否支持GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 创建一个随机张量
tensor = torch.rand(2, 2)
print(tensor)

print("Device tensor is stored on:", tensor.device)

# 将张量移动到GPU上
if torch.cuda.is_available():
    tensor = tensor.to(device)
    print(tensor)   

print("Device tensor is stored on:", tensor.device) 
