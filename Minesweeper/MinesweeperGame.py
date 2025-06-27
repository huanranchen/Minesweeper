import random

__all__ = ["Minesweeper"]


class Minesweeper:
    ROWS, COLS = 10, 10

    MINES = 10

    grid = []
    revealed = []
    flagged = []
    game_over = False

    def initialize_game(self):
        self.grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.revealed = [[False for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.flagged = [[False for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.game_over = False

        mine_positions = random.sample([(i, j) for i in range(self.ROWS) for j in range(self.COLS)], self.MINES)
        for i, j in mine_positions:
            self.grid[i][j] = -1

        for i in range(self.ROWS):
            for j in range(self.COLS):
                if self.grid[i][j] == -1:
                    continue
                count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.ROWS and 0 <= nj < self.COLS and self.grid[ni][nj] == -1:
                            count += 1
                self.grid[i][j] = count

    def reveal_cell(self, i, j):
        if i < 0 or i >= self.ROWS or j < 0 or j >= self.COLS or self.revealed[i][j] or self.flagged[i][j]:
            return
        self.revealed[i][j] = True
        if self.grid[i][j] == -1 or self.check_win():
            self.game_over = True
        elif self.grid[i][j] == 0:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    self.reveal_cell(i + di, j + dj)

    def check_win(self) -> bool:
        if self.ROWS * self.COLS - self.MINES == len(self.revealed):
            return True
        return False

    def handle_click(self, i: int, j: int, right_click=False):
        if self.game_over:
            return
        if i < 0 or i >= self.ROWS or j < 0 or j >= self.COLS:
            return
        if right_click:
            if not self.revealed[i][j]:
                self.flagged[i][j] = not self.flagged[i][j]
        else:
            if not self.flagged[i][j]:
                self.reveal_cell(i, j)
