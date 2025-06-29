import torch
import pygame
from models import get_bert_like_01_out_transformer
import asyncio
import platform
from Minesweeper import *

model = get_bert_like_01_out_transformer()
model.load_state_dict(torch.load("checkpoints/model_1.pt", map_location=torch.device('cpu')))
model.eval()

gamer = MinesweeperVisualizer(Minesweeper())


async def update_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False

    model.next_move(gamer.minesweeper, execute=True)
    gamer.minesweeper.check_win(verbose=True)
    gamer.draw()
    if gamer.minesweeper.game_over:
        return False
    return True


async def main():
    gamer.ui_setup()
    step = 1
    while True:
        if not await update_loop():
            break
        print(step)
        await asyncio.sleep(3.0)
        step += 1


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
