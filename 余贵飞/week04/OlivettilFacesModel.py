# encoding: utf-8
# @File  : OlivettilFacesModel.py
# @Author: GUIFEI
# @Desc : 定义不同 olivetti faces 模型
# @Date  :  2025/03/20
import BaseModel

# 使用 BatchNorm1d 归一化，不使用正则, 使用 ReLU 激活函数
class OlivettiFaces1(BaseModel.BaseModel):
    def __init__(self):
        super().__init__(norm_type = "batch")

# 使用 LayerNorm 归一化，不使用正则, 使用 ReLU 激活函数
class OlivettiFaces2(BaseModel.BaseModel):
    def __init__(self):
        super().__init__(norm_type = "layer")


# 不使用，使用正则Dropout(0.3), 使用 ReLU 激活函数
class OlivettiFaces3(BaseModel.BaseModel):
    def __init__(self):
        super().__init__(dropout_rate = 0.3)

# 不使用，使用正则Dropout(0.5), 使用 ReLU 激活函数
class OlivettiFaces4(BaseModel.BaseModel):
    def __init__(self):
        super().__init__(dropout_rate = 0.5)

# 同时使用BatchNorm1d归一化及正则Dropout(0.4)
class OlivettiFaces5(BaseModel.BaseModel):
    def __init__(self):
        super().__init__(norm_type = "batch", dropout_rate = 0.4)


