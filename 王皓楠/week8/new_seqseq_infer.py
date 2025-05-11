import torch
import pickle
from new_seqseq import Seq2Seq

if __name__ == '__main__':
    # 加载训练好的模型和词典
    state_dict = torch.load('seq2seq_state.bin')
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        mod=0,
        dropout=0.5,
    )
    model.load_state_dict(state_dict)

    
    dvoc_inv = {v:k for k,v in dvoc.items()}

    # 输入
    enc_input = "一剑光寒十四州"
    enc_idx = torch.tensor([[evoc[tk] for tk in enc_input.split()]])

    print(enc_idx.shape)

    max_dec_len = 50

    model.eval()
    with torch.no_grad():
        #由于加入attention因此需要修改
        
        hidden_state,outputs = model.encoder(enc_idx)
       

       
        dec_input = torch.tensor([[dvoc['BOS']]])

        # 循环decoder
        dec_tokens = []
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
          
            logits,hidden_state = model.decoder(dec_input, hidden_state,outputs)
          
            
           
            next_token = torch.argmax(logits, dim=-1)

            if dvoc_inv[next_token.squeeze().item()] == 'EOS':
                break
         
            dec_tokens.append(next_token.squeeze().item())
            
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)

    # 输出解码结果
    print(''.join([dvoc_inv[tk] for tk in dec_tokens]))