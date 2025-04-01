import pandas as pd
import numpy as np

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 生成模拟特征数据
num_samples = 100
data = {
    "平均半径": np.round(np.random.uniform(10, 25, num_samples), 2),
    "平均纹理": np.round(np.random.uniform(15, 35, num_samples), 2),
    "平均周长": np.round(np.random.uniform(70, 180, num_samples), 2),
    "平均面积": np.round(np.random.uniform(400, 1800, num_samples), 2),
    "平均平滑度": np.round(np.random.uniform(0.08, 0.15, num_samples), 2),
    "平均紧凑度": np.round(np.random.uniform(0.05, 0.3, num_samples), 2),
    # 添加更多特征...
}

# 生成模拟目标标签（0：良性，1：恶性）
# 这里我们简单地根据 "平均半径" 生成目标标签
# 实际应用中，目标标签应根据真实数据生成
data["诊断结果"] = np.where(data["平均半径"] > 18, 1, 0)

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("党金虎/week02/breast_cancer_sample_data_chinese.csv", index=False, encoding="utf-8-sig")

print("CSV 样例数据已生成：breast_cancer_sample_data_chinese.csv")