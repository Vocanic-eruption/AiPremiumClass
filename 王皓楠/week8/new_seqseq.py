import torch
import torch.nn as nn

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, mod,dropout):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hidden_dim,dropout=dropout, 
                          batch_first=True, bidirectional=True)
        self.mod=mod

    def forward(self, token_seq):
        
        embedded = self.embedding(token_seq)
       
        outputs, hidden = self.rnn(embedded)
       
     
        if self.mod ==0:
            return torch.cat((hidden[0], hidden[1]), dim=1),outputs
       
        return hidden.sum(dim=0),outputs

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output, dec_output):
        # a_t = h_t @ h_s  
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 1.计算 结合解码token和编码token，关联的权重
        a_t = torch.softmax(a_t, dim=1)
        # 2.计算 关联权重和编码token 贡献值
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t
    
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, mod,dropout):
        super(Decoder, self).__init__()
        self.hiddens= hidden_dim*2 if mod ==0 else hidden_dim
       
        self.embedding = nn.Embedding(input_dim, emb_dim)
       
        self.rnn = nn.GRU(emb_dim, self.hiddens, dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(self.hiddens, input_dim)  # 解码词典中词汇概率
        self.attention=Attention()

    def forward(self, token_seq, hidden_state, enc_output):
        embedded = self.embedding(token_seq)
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))
        c_t = self.attention(enc_output, dec_output)
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        out = torch.tanh(self.atteniton_fc(cat_output))
        logits = self.fc(out)
        return logits, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self,
                 enc_emb_size, 
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 mod,
                 dropout=0.5,
                 ):
        
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, mod,dropout=dropout)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, mod,dropout=dropout)


    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state, outputs = self.encoder(enc_input)
        output,hidden = self.decoder(dec_input, encoder_state, outputs)
        return output,hidden

if __name__ == '__main__':
    
    # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10

  

    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        mod =0,
        dropout=dropout,
    )
    seqseq2=Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim, 
        mod =1,
        dropout=dropout,
    )
    logits = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    logits2 = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape) 
    print(logits2.shape)

