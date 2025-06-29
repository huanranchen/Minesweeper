import torch
from models import get_bert_like_01_out_transformer
from trainer import PNLTrainer

model = get_bert_like_01_out_transformer()
model.load_state_dict(torch.load("./checkpoints/model.pt"))
trainer = PNLTrainer(dict(), model)
trainer.train()
