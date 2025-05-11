import torch
# from EncoderDecoderModel import Seq2Seq
from EncoderDecoderAttentionModel import Seq2Seq
import pickle
from data_process import Vocabulary

"""
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""

if __name__ == '__main__':
    model_state = torch.load('seq2seq_state_attention.bin')
    # model_state = torch.load('seq2seq_state.bin')
    with open('vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    model = Seq2Seq(
        len(enc_vocab.vocab),
        len(dec_vocab.vocab),
        150,
        128,
        0.5)
    model.load_state_dict(model_state)

    # 创建解码器反向解码
    rever_vocab = {idx: word for word, idx in dec_vocab.vocab.items()}

    user_input = 'Hi'
    # user_input = "What I'm about to say is strictly between you and me"
    split = user_input.split()
    enc_inputs = torch.tensor([[enc_vocab.vocab[word] for word in split]])
    # [batch_size,seq_len]
    # print(enc_inputs.shape)

    model.eval()
    with torch.no_grad():
        """
        - 用户输入通过vocab转换token_index
        - token_index通过encoder获取 encoder last hidden_state
        - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
        - 循环decoder
            - decoder输入:[['BOS']], hidden_state
            - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
            - 计算argmax, 的到下一个token_index
            - decoder的下一个输入 = token_index
            - 收集每次token_index 【解码集合】
        - 输出解码结果
        """
        # last_hidden = model.encoder(enc_inputs, 'cat')
        last_hidden, enc_outputs = model.encoder(enc_inputs, 'cat')
        max_len = 50

        dec_inputs = torch.tensor([[dec_vocab.vocab['BOS']]])
        dec_outputs = []
        while True:
            if len(dec_outputs) >= max_len:
                break
            # logits, last_hidden = model.decoder(dec_inputs, last_hidden)
            logits, last_hidden = model.decoder(dec_inputs, last_hidden, enc_outputs)

            next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == dec_vocab.vocab['EOS']:
                break
            dec_outputs.append(next_token.squeeze().item())
            dec_inputs = next_token

    print(''.join([rever_vocab[idx] for idx in dec_outputs]))
