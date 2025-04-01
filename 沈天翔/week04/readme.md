1. 搭建的神经网络，使用olivettiface数据集进行训练。
fetch_olivetti_faces在scikit-learn 1.2版本后已被弃用移除，因此用 fetch_openml 替代

2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
3. 尝试不同optimizer对模型进行训练，观察对比loss结果。
