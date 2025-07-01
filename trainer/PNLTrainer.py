import os
import torch
import wandb
from tqdm import tqdm
from torch import nn
from Minesweeper import Minesweeper
from typing import List

__all__ = ["PNLTrainer"]


class PNLTrainer:
    """
    Positive Negative Loss Trainer
    """

    def __init__(
        self,
        minesweeper_cfg: dict,
        model: nn.Module,
        wandb_name: str = "debug",
        lr: float = 1e-4,
        device=torch.device("cuda"),
    ):
        self.minesweeper_cfg = minesweeper_cfg
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        self.model.to(self.device)
        wandb_name = wandb_name + "-lr=" + str(lr)
        wandb.init(project="minesweeper", name=wandb_name)
        self.wandb_name = wandb_name
        self.ckpt_path = os.path.join("./checkpoints", self.wandb_name)
        os.makedirs(self.ckpt_path, exist_ok=True)

    def prepare_input(self, minesweeper: Minesweeper):
        grid = torch.tensor(minesweeper.grid, device=self.device).reshape(-1)
        revealed = torch.tensor(minesweeper.revealed, device=self.device).reshape(-1)
        x = grid.clone()
        x[~revealed] = 9  # unknown
        y = grid.clone()
        mine_position = grid == -1
        y[revealed] = -1  # 雷和revealed是-1
        y[y >= 0] = 0  # 其他都可以点
        y += 1  # 雷和revealed是0， 其他（正常数字和空）是1
        # loss weight
        loss_weight = torch.zeros_like(y)  # 目前改成了只计算有雷的和有数字的loss
        loss_weight[mine_position] = 10  # 雷的地方给10倍的loss
        loss_weight[y == 1] = 2
        return x, y, loss_weight, revealed  # 都是L 的tensor

    def prepare_input_batch(self, minesweeper_list: List[Minesweeper]):
        results = [self.prepare_input(i) for i in minesweeper_list]
        xs, ys, w, revealed = (
            torch.stack([i[0] for i in results]),
            torch.stack([i[1] for i in results]),
            torch.stack([i[2] for i in results]),
            torch.stack([i[3] for i in results]),
        )
        return xs, ys, w, revealed  # 都是(B, L)的tensor

    def train(
        self,
        batch_size: int = 512,  # effective batch size here
        total_iter: int = 131072,
        pool_update_frequency: int = 10,
        saving_frequency: int = 10,
        gradient_accumulation_steps: int = 1,
    ):
        if os.path.exists(os.path.join(self.ckpt_path, "model.pt")):
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, "model.pt"), map_location=self.device))
        total_batch_size = batch_size
        batch_size = total_batch_size // gradient_accumulation_steps
        assert saving_frequency % pool_update_frequency == 0
        assert total_batch_size % gradient_accumulation_steps == 0
        games = [Minesweeper(**self.minesweeper_cfg) for _ in range(total_batch_size)]
        self.model.train()
        for num_iter in tqdm(range(1, total_iter + 1)):
            if num_iter % 5000 == 0:
                pool_update_frequency = max(pool_update_frequency // 2, 1 if num_iter > 65536 else 2)
                print(f"pool update frequency: {pool_update_frequency}")
            games = [i for i in games if not i.game_over]
            games = games + [Minesweeper(**self.minesweeper_cfg) for _ in range(total_batch_size - len(games))]
            self.optimizer.zero_grad()
            all_logits, all_revealed = [], []
            for accumulation in range(0, total_batch_size, batch_size):
                x, y, w, revealed = self.prepare_input_batch(games[accumulation : accumulation + batch_size])
                logits = self.model(x).logits  # B, L, 2
                all_logits.append(logits)
                all_revealed.append(revealed)
                loss = self.criterion(logits.view(-1, 2), y.view(-1)) * w.view(-1)
                loss = loss.mean()
                loss.backward()
            self.optimizer.step()
            logits, revealed = torch.cat(all_logits, dim=0), torch.cat(all_revealed, dim=0)
            if num_iter % pool_update_frequency == 0:
                # B, L, 2
                logits = torch.softmax(logits, dim=-1)  # 归一化后再选最大的，防止出事
                # 只取预测为1的位置，0的不取。但是加上这行后有可能全都无法预测，导致无任何进展
                # logits[logits[:, :, 0] > logits[:, :, 1]] = float("-inf")
                logits[revealed] = float("-inf")  # 已经开了的不取
                prediction = logits[:, :, 1].argmax(dim=1)  # B
                row, col = prediction // games[0].COLS, prediction % games[0].COLS
                row, col = row.tolist(), col.tolist()
                for i, game in enumerate(games):
                    game.handle_click(row[i], col[i])
                progress = [sum([sum(i) for i in game.revealed]) for game in games]
                # print(progress[:10])
                # import pdb
                #
                # pdb.set_trace()

            if num_iter % saving_frequency == 0:
                num_game_over = sum([game.game_over for game in games])
                num_win = sum([game.check_win() for game in games])
                num_lost = num_game_over - num_win
                single_step_acc = (total_batch_size - num_lost) / total_batch_size  # Single step acc. Approach 100%
                avg_steps = sum(progress) / total_batch_size
                # Overall acc.
                loss = round(loss.item(), 4)
                acc = round(num_win / num_game_over, 4)
                print(f"loss={loss}, acc={acc}, avg steps={round(avg_steps, 4)}")
                print(f"num_game_over {num_game_over}, num_wins {num_win}")
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "model.pt"))
                wandb.log(
                    {
                        "optim/loss": loss,
                        "eval/all_step_acc": acc,
                        "eval/single_step_acc": single_step_acc,
                        "eval/avg_steps": avg_steps,
                        "eval/num_game_over": num_game_over,
                        "eval/num_win": num_win,
                        "misc/pool_update_frequency": pool_update_frequency,
                        "optim/lr": self.optimizer.param_groups[0]["lr"],
                        "optim/batch_size": total_batch_size,
                    },
                    step=num_iter,
                )
