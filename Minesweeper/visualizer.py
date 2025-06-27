import pygame
from .MinesweeperGame import Minesweeper


__all__ = ["MinesweeperVisualizer"]


class MinesweeperVisualizer:
    WIDTH, HEIGHT = 400, 400
    FPS = 60
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (192, 192, 192)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    def __init__(self, minesweeper: Minesweeper):
        self.minesweeper = minesweeper
        self.CELL_SIZE = self.WIDTH // self.minesweeper.COLS

    def ui_setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Minesweeper")
        self.font = pygame.font.Font(None, 36)
        self.minesweeper.initialize_game()

    def draw(self):
        self.screen.fill(self.WHITE)
        for i in range(self.minesweeper.ROWS):
            for j in range(self.minesweeper.COLS):
                rect = pygame.Rect(j * self.CELL_SIZE, i * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.minesweeper.revealed[i][j]:
                    pygame.draw.rect(self.screen, self.GRAY, rect)
                    if self.minesweeper.grid[i][j] == -1:
                        pygame.draw.circle(self.screen, self.RED, rect.center, self.CELL_SIZE // 4)
                    elif self.minesweeper.grid[i][j] > 0:
                        text = self.font.render(str(self.minesweeper.grid[i][j]), True, self.BLUE)
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.GRAY, rect)
                    if self.minesweeper.flagged[i][j]:
                        text = self.font.render("F", True, self.BLACK)
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        if self.minesweeper.game_over:
            if self.minesweeper.check_win():
                text = self.font.render("Win!", True, self.RED)
            else:
                text = self.font.render("Game Over!", True, self.RED)
            text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text, text_rect)
        pygame.display.flip()

    def handle_click(self, pos, right_click=False):
        return self.minesweeper.handle_click(pos[1] // self.CELL_SIZE, pos[0] // self.CELL_SIZE, right_click)
