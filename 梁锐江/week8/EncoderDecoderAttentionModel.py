import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, dropout=dropout_rate, batch_first=True, bidirectional=True)

    def forward(self, inputs, type):
        embedded = self.embedding(inputs)
        outputs, hidden = self.rnn(embedded)
        # return hidden
        states = hidden.split(1, dim=0)
        if type == 'cat':
            return torch.cat(states, dim=-1), outputs
        # elif type == 'add':
        #     return torch.add(states[0], states[1])
        # else:
        #     return torch.mul(states[0], states[1])


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim * 2, dropout=dropout_rate, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

        self.attention = Attention()
        self.attention_linear = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def forward(self, inputs, hidden, enc_outputs):
        embedded = self.embedding(inputs)
        # decoder_hidden = hidden.repeat(4, 1, 1)  # 复制隐藏状态
        dec_outputs, hidden = self.rnn(embedded, hidden)

        c_t = self.attention(enc_outputs, dec_outputs)
        out = torch.tanh(self.attention_linear(torch.cat((c_t, dec_outputs), dim=-1)))

        logits = self.linear(out)
        return logits, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, enc_outputs, dec_outputs):
        s_t = torch.bmm(enc_outputs, dec_outputs.permute(0, 2, 1))
        a_t = torch.softmax(s_t, dim=-1)
        return torch.bmm(a_t.permute(0, 2, 1), enc_outputs)


class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(enc_vocab_size, embedding_dim, hidden_dim, dropout_rate)
        self.decoder = Decoder(dec_vocab_size, embedding_dim, hidden_dim, dropout_rate)

    def forward(self, enc_inputs, dec_inputs):
        encoder_output, enc_outputs = self.encoder(enc_inputs, 'cat')
        decoder_output, decoder_hidden = self.decoder(dec_inputs, encoder_output, enc_outputs)
        return decoder_output, decoder_hidden


if __name__ == '__main__':
    enc_vocab_size = 100
    dec_vocab_size = 105
    embedding_dim = 120
    hidden_dim = 110
    dropout_rate = 0.5
    batch_size = 5000
    seq_len = 20

    seq = Seq2Seq(enc_vocab_size, dec_vocab_size, embedding_dim, hidden_dim, dropout_rate)

    outputs, hidden = seq(torch.randint(0, enc_vocab_size, (batch_size, seq_len)),
                          torch.randint(0, dec_vocab_size, (batch_size, seq_len)))
    print(outputs.shape)
