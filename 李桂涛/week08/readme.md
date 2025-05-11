### seq2seq
**数据预处理部分**：

- 加载词汇表，处理输入输出文件，构建词到索引的映射。

- 数据填充（padding）和数据集创建，使用PyTorch的DataLoader。

**模型构建部分**：

- Encoder：使用单向LSTM，可能需要支持不同的隐藏状态合并方式。

- Attention机制：Bahdanau Attention的实现。

- Decoder：在解码过程中集成Attention。

**训练循环**：

- 损失函数和优化器的定义。

- 训练步骤的实现，包括前向传播、损失计算、反向传播。

- 集成TensorBoard进行训练监控。

**推理实现**：

- 实现生成对联的函数，使用训练好的模型进行预测。
