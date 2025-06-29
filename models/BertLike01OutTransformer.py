import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch import nn
from Minesweeper import Minesweeper

__all__ = [
    "get_bert_like_01_out_transformer",
    "get_bert_like_01_out_transformer_gpt2_large",
    "get_bert_like_01_out_transformer_gpt2_middle",
    "get_bert_like_01_out_transformer_gpt2_small",
    "get_bert_like_01_out_transformer_gpt2_xl",
]

"""
约定输入：
0-8表示数字
9表示unknown

约定输出：
0 不点击
1 点击
"""


class BertLike01OutTransformer(GPT2LMHeadModel):
    def __init__(self, config, output_vocab_size: int = 2):
        super().__init__(config)  # 我这里重定义了lm_head
        self.lm_head = nn.Linear(config.n_embd, output_vocab_size, bias=False)

    def prepare_input(self, minesweeper: Minesweeper):
        grid = torch.tensor(minesweeper.grid, device=self.device).reshape(-1)
        revealed = torch.tensor(minesweeper.revealed, device=self.device).reshape(-1)
        x = grid.clone()
        x[~revealed] = 9  # unknown
        y = grid.clone()
        y[revealed] = -1  # 雷和revealed是-1
        y[y > 0] = 0  # 其他都可以点
        y += 1  # 雷和revealed是0， 其他是1
        return x, y, revealed  # x, y都是L 的tensor

    def next_move(self, minesweeper: Minesweeper, execute: bool = False):
        x, _, revealed = self.prepare_input(minesweeper)
        x = x.unsqueeze(0)  # 1, L
        logits = self(x).logits  # 1, L, 2
        logits[logits[:, :, 0] > logits[:, :, 1]] = float(
            "-inf"
        )  # 只取预测为1的位置。预测为0的不取  # TODO：参考训练，这里可能要归一化后找。而且不一定要排除0>1的。（否则可能stuck）
        logits = logits.squeeze()  # L, 2
        logits[revealed] = float("-inf")  # 已经开了的格子不取  TODO: 训练那块需要这样做吗
        prediction = logits[:, 1].argmax()  # scalar
        row, col = prediction // minesweeper.COLS, prediction % minesweeper.COLS
        row, col = row.item(), col.item()
        if execute:
            minesweeper.handle_click(row, col)
        return row, col


def get_bert_like_01_out_transformer(
    n_positions: int = 1024,
    n_embd: int = 768,
    n_layer: int = 6,
    n_head: int = 4,
    vocab_size: int = 10,
):
    model = BertLike01OutTransformer(
        GPT2Config(
            vocab_size=vocab_size,
            n_layers=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            n_embd=n_embd,
        )
    )
    return model


def get_bert_like_01_out_transformer_gpt2_small(
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    *args,
    **kwargs,
):
    return get_bert_like_01_out_transformer(n_embd=n_embd, n_layer=n_layer, n_head=n_head, *args, **kwargs)


def get_bert_like_01_out_transformer_gpt2_middle(
    n_embd: int = 1024,
    n_layer: int = 24,
    n_head: int = 16,
    *args,
    **kwargs,
):
    return get_bert_like_01_out_transformer(n_embd=n_embd, n_layer=n_layer, n_head=n_head, *args, **kwargs)


def get_bert_like_01_out_transformer_gpt2_large(
    n_embd: int = 1280,
    n_layer: int = 36,
    n_head: int = 20,
    *args,
    **kwargs,
):
    return get_bert_like_01_out_transformer(n_embd=n_embd, n_layer=n_layer, n_head=n_head, *args, **kwargs)


def get_bert_like_01_out_transformer_gpt2_xl(
    n_embd: int = 1600,
    n_layer: int = 48,
    n_head: int = 25,
    *args,
    **kwargs,
):
    return get_bert_like_01_out_transformer(n_embd=n_embd, n_layer=n_layer, n_head=n_head, *args, **kwargs)
