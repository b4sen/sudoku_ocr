import numpy as np

class Solver:

    def __init__(self):
        self.board = None
    
    def set_board(self, board):
        self.board = board
    
    def get_board(self):
        return self.board

    # finds next empty value
    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return (i, j)  # row, col

        return False

    # checks whether given num for pos (list  [x, y]) on board
    def is_valid(self, num, pos):
        # Check row
        for i in range(len(self.board[0])):
            if self.board[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self.board[i][j] == num and (i, j) != pos:
                    return False

        return True


    def solve(self):
        # find next empty position:
        pos = self.find_empty()
        if pos:
            for num in range(1, 10):
                # if num is suitable candidate for empty position, put it there, and solve the less complex board
                if self.is_valid(num, pos):
                    self.board[pos[0]][pos[1]] = num

                    # if we can proceed with filling, we do so
                    if self.solve():
                        return True
                    # if we cant proceed we reset the field
                    else:
                        self.board[pos[0]][pos[1]] = 0
        else:
            # in this case we did not find any empty places left
            return True


if __name__ == '__main__':
    import timeit
    board = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7]
    ]

    solver = Solver()
    solver.set_board(board)
    solver.solve()
    print(np.matrix(solver.get_board()))