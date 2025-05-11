import pickle
import torch
import json
from torch.utils.data import DataLoader
from couplet_seq2seq import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from couplet_process import batch_proc_factory
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('vocab.bin', 'rb') as f:
        evoc, dvoc = pickle.load(f)
    with open('encoder.json', 'r', encoding='utf-8') as f:
        enc_data = json.load(f)
    with open('decoder.json', 'r', encoding='utf-8') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data, dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=batch_proc_factory(evoc, dvoc))
    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0,
    )
    model.to(device)
    writer = SummaryWriter(log_dir='./runs/seq2seq')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(20):
        model.train()
        total_loss = 0
        tpbar = tqdm(dl)
        for batch_idx, (enc_input, dec_input, targets) in enumerate(tpbar):
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)
            logits, _ = model(enc_input, dec_input)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dl) + batch_idx)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tpbar.set_description(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(dl)
        writer.add_scalar('Avg Loss/train', avg_loss, epoch)
    writer.close()
    torch.save(model.state_dict(), 'seq2seq_state.bin')