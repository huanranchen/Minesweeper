import torch
from models import get_bert_like_01_out_transformer_gpt2_middle
from trainer import PNLTrainer

model = get_bert_like_01_out_transformer_gpt2_middle()
# model.load_state_dict(torch.load("./checkpoints/model.pt"))
trainer = PNLTrainer(dict(), model, "middle")
trainer.train(gradient_accumulation_steps=2)
