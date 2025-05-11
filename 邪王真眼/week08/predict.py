import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import plugins as plugins

from models.models import get_model
from datasets.datasets import get_dataset
from configs.config import load_train_config


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week08.py")
    model_path = "./邪王真眼/week08/result/simple_attention.pth"
    cfg['batch_size'] = 1

    # dataset
    _, val_loader = get_dataset(cfg)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # val
    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            label = batch_data.pop("label")

            outputs = torch.zeros((batch_data['x_decode'].shape[0], cfg.get('vocab_size'), batch_data['x_decode'].shape[1])).to(device)
            for i in range(batch_data['x_decode'].shape[1]):
                step_output = model(**batch_data)
                outputs[:, :, i] = step_output['out'][:, :, i]

            pred_ids = outputs.argmax(dim=1).squeeze(0)
            print("输入上联: ",val_loader.dataset.decode(batch_data["x"][0]))
            print("真实下联: ", val_loader.dataset.decode(label[0][:-1]))
            print("预测下联: ", val_loader.dataset.decode(pred_ids[:-1]))
            print("-" * 70)
            input()
