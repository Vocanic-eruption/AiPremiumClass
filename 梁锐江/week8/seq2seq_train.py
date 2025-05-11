import json
import pickle
from torch.utils.data import DataLoader
from data_process import Vocabulary, BatchDataProcessor
from EncoderDecoderAttentionModel import Seq2Seq
# from EncoderDecoderModel import Seq2Seq
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    epochs = 15

    writer = SummaryWriter('exercise_attention')

    with open('encode.json', 'r', encoding='utf-8') as f:
        enc_doc = json.load(f)

    with open('decoder.json', 'r', encoding='utf-8') as f:
        dec_doc = json.load(f)

    with open('vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    batch_processor = BatchDataProcessor(enc_vocab.vocab, dec_vocab.vocab)
    dataset = list(zip(enc_doc, dec_doc))
    dl = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=batch_processor.get_word_idx)

    model = Seq2Seq(len(enc_vocab.vocab), len(dec_vocab.vocab), 150, 128, 0.5)
    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_step = 0
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, targets in dl:
            model.train()
            enc_inputs = enc_inputs.to('cuda')
            dec_inputs = dec_inputs.to('cuda')
            targets = targets.to('cuda')

            optimizer.zero_grad()
            # forward.shape ['batch_size','seq_len','vocab.size']
            logits, _ = model(enc_inputs, dec_inputs)

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_step += 1
            writer.add_scalar('loss', loss, total_step)
        print(f"{epoch} running")
    writer.close()

    torch.save(model.state_dict(), 'seq2seq_state_attention.bin')
