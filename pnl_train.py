import torch
from models import get_bert_like_01_out_transformer_gpt2_small
from trainer import PNLTrainer

model = get_bert_like_01_out_transformer_gpt2_small()
# model.load_state_dict(torch.load("./checkpoints/model.pt"))
trainer = PNLTrainer(dict(), model, "small")
trainer.train()
