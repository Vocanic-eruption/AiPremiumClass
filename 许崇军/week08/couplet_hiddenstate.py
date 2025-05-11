
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout,model='cat'):
        super(Encoder, self).__init__()
        self.model = model
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=0,
                          batch_first=True, bidirectional=True)



    def forward(self, token_seq):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        if self.model == 'cat':
            final_hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        elif self.model == 'add':
             final_hidden = torch.add(hidden[0], hidden[1])
        else:
            raise ValueError("Unsupported mode, use 'concat' or 'add'")
        return final_hidden, outputs



class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output, dec_output):
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        a_t = torch.softmax(a_t, dim=1)
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t

class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.atteniton = Attention()
        self.atteniton_fc = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def forward(self, token_seq, hidden_state, enc_output):
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [1, batch_size, hidden_dim * 2]
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))
        c_t = self.atteniton(enc_output, dec_output)
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        out = torch.tanh(self.atteniton_fc(cat_output))
        # out: [batch_size, seq_len, hidden_dim * 2]
        logits = self.fc(out)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self,
                 enc_emb_size,
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5,
                 ):
        super().__init__()

        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout=dropout)
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout=dropout)

    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state, outputs = self.encoder(enc_input)
        output, hidden = self.decoder(dec_input, encoder_state, outputs)

        return output, hidden


if __name__ == '__main__':
    # 测试Encoder
    input_dim = 1345
    emb_dim = 256
    hidden_dim = 128
    dropout = 0.5
    batch_size = 4
    seq_len = 103

    encoder_cat = Encoder(input_dim, emb_dim, hidden_dim, dropout,model='cat')
    encoder_add = Encoder(input_dim, emb_dim, hidden_dim, dropout,model='add')


    token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    hidden_state_cat, outputs_cat = encoder_cat(token_seq)  # Encoder输出（最后时间步状态）
    token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    hidden_state_add, outputs_add = encoder_add(token_seq)  # Encoder输出（最后时间步状态）
    print("encoder的输出shape为",hidden_state_cat.shape)  # 应该是 [batch_size, hidden_dim]
    print("encoder的输出shape为",hidden_state_add.shape)  # 应该是 [batch_size, hidden_dim]

'''
双向GRU的隐藏层结构
hidden_dim = 128  
bidirectional = True  # 双向GRU
cat模式（拼接）：
final_hidden = torch.cat((hidden[0], hidden[1]), dim=1)  
# hidden[0]和hidden[1]都是 [4, 128]，拼接后得到 [4, 256]
add模式（相加）：
final_hidden = torch.add(hidden[0], hidden[1])  
# 两个 [4, 128] 张量相加，保持维度为 [4, 128]
'''

# for model in ['cat', 'add']:
#     encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout, model=model)
#     token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
#     hidden_state, outputs = encoder(token_seq)  # Encoder输出（最后时间步状态）
#     print(f"encoder的输出shape为{model}：",hidden_state.shape)  # 应该是 [batch_size, hidden_dim]



    #
    # # 测试Decoder
    # decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # dec_logits,_ = decoder(token_seq, hidden_state, outputs)  # Decoder输出
    # print("decoder的输出shape为", dec_logits.shape)  # 应该是 [batch_size, seq_len, input_dim]
    #
    # seq2seq = Seq2Seq(
    #     enc_emb_size=input_dim,
    #     dec_emb_size=input_dim,
    #     emb_dim=emb_dim,
    #     hidden_size=hidden_dim,
    #     dropout=dropout
    # )
    #
    # logits, _ = seq2seq(
    #     enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
    #     dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    # )
    # print("ses2seq的输出shape为", logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

