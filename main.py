import pygame
import asyncio
import platform
from Minesweeper import *


gamer = MinesweeperVisualizer(Minesweeper())


def update_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
        elif event.type == pygame.MOUSEBUTTONDOWN:
            gamer.handle_click(event.pos, event.button == 3)  # 右键是 button 3
    gamer.draw()


async def main():
    gamer.ui_setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / gamer.FPS)


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
