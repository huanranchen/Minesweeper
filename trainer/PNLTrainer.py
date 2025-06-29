import os
import torch
from tqdm import tqdm
from torch import nn
from Minesweeper import Minesweeper
from typing import List

__all__ = ["PNLTrainer"]


class PNLTrainer:
    """
    Positive Negative Loss Trainer
    """

    def __init__(self, minesweeper_cfg: dict, model: nn.Module, device=torch.device("cuda")):
        self.minesweeper_cfg = minesweeper_cfg
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4, weight_decay=0)
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        self.model.to(self.device)

    def prepare_input(self, minesweeper: Minesweeper):
        grid = torch.tensor(minesweeper.grid, device=self.device).reshape(-1)
        revealed = torch.tensor(minesweeper.revealed, device=self.device).reshape(-1)
        x = grid.clone()
        x[~revealed] = 9  # unknown
        y = grid.clone()
        mine_position = grid == -1
        y[revealed] = -1  # 雷和revealed是-1
        y[y > 0] = 0  # 其他都可以点
        y += 1  # 雷和revealed是0， 其他是1
        # loss weight
        loss_weight = torch.ones_like(y)
        loss_weight[mine_position] = 10  # 雷的地方给10倍的loss
        return x, y, loss_weight  # 都是L 的tensor

    def prepare_input_batch(self, minesweeper_list: List[Minesweeper]):
        results = [self.prepare_input(i) for i in minesweeper_list]
        xs, ys, w = (
            torch.stack([i[0] for i in results]),
            torch.stack([i[1] for i in results]),
            torch.stack([i[2] for i in results]),
        )
        return xs, ys, w  # 都是(B, L)的tensor

    def train(self, batch_size: int = 256, total_iter: int = 10240, pool_update_frequency: int = 10):
        games = [Minesweeper(**self.minesweeper_cfg) for _ in range(batch_size)]
        epoch_loss = 0
        self.model.train()
        for num_iter in tqdm(range(1, total_iter + 1)):
            games = [i for i in games if not i.game_over]
            games = games + [Minesweeper(**self.minesweeper_cfg) for _ in range(batch_size - len(games))]
            x, y, w = self.prepare_input_batch(games)
            logits = self.model(x).logits  # B, L, 2
            loss = self.criterion(logits.view(-1, 2), y.view(-1)) * w.view(-1)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            if num_iter % pool_update_frequency == 0:
                logits[logits[:, :, 0] > logits[:, :, 1]] = float("-inf")  # 只取预测为1的位置  # B, L, 2
                prediction = logits[:, :, 1].argmax(dim=1)  # B
                row, col = prediction // games[0].COLS, prediction % games[0].COLS
                row, col = row.tolist(), col.tolist()
                for i, game in enumerate(games):
                    game.handle_click(row[i], col[i])
                num_lost = sum([game.game_over for game in games]) - sum([game.check_win() for game in games])
                cur_acc = (batch_size - num_lost) / batch_size
                avg_steps = sum([sum([sum(i) for i in game.revealed]) for game in games]) / batch_size
                print(f"loss={loss.item()}, acc={cur_acc}, avg steps={avg_steps}")
                torch.save(self.model.state_dict(), "./checkpoints/model.pt")
