import torch
import pickle
from couplet_seq2seq import Encoder, Decoder, Seq2Seq
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('vocab.bin', 'rb') as f:
        evoc, dvoc = pickle.load(f)
    dvoc_inv = {v: k for k, v in dvoc.items()}

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0,
    ).to(device)
    model.load_state_dict(torch.load('seq2seq_state.bin', map_location=device))

    model.eval()
    while True:
        input_str = input("请输入上联（输入q退出）：")
        if input_str == 'q':
            break

        input_clean = input_str.replace(' ', '')
        enc_idx = torch.tensor([[evoc[tk] for tk in input_clean]], device=device)
        with torch.no_grad():
            hidden_state, enc_outputs = model.encoder(enc_idx)
            dec_input = torch.tensor([[dvoc['BOS']]], device=device)
            dec_tokens = []
            for _ in range(20):
                logits, hidden_state = model.decoder(
                    dec_input,
                    hidden_state,
                    enc_outputs
                )
                next_token = logits.argmax(-1)
                if dvoc_inv[next_token.item()] == 'EOS':
                    break

                dec_tokens.append(next_token.item())
                dec_input = next_token
                hidden_state = hidden_state.squeeze(0)

        result = ''.join([dvoc_inv[tk] for tk in dec_tokens])
        print("下联预测结果：", result)