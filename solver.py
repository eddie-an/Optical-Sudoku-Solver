def is_valid_sudoku(board) -> bool:
    if (board == None or len(board) != 9 or len(board[0]) != 9):
        return False
    filledRows = [set() for _ in range(9)]
    filledCols = [set() for _ in range(9)]
    filledBoxes = [set() for _ in range(9)]

    for i in range(9):
        for j in range(9):
            boxNum = (i // 3) * 3 + (j // 3)

            if board[i][j] != ".":
                digit = int(board[i][j])
                if ((digit not in filledRows[i]) and 
                    (digit not in filledCols[j]) and 
                    (digit not in filledBoxes[boxNum])):
                    filledRows[i].add(digit)
                    filledCols[j].add(digit)
                    filledBoxes[boxNum].add(digit)
                else:
                    return False
    return True

def solve_sudoku(board):
    filledRows = [set() for _ in range(9)]
    filledCols = [set() for _ in range(9)]
    filledBoxes = [set() for _ in range(9)]

    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                k = int(board[i][j])
                filledRows[i].add(k)
                filledCols[j].add(k)
                key = i//3 * 3 + j//3
                filledBoxes[key].add(k)

    def _backtrack(r, c):
        if r == 9:
            return True

        new_r = r + (c + 1) // 9
        new_c = (c + 1) % 9

        if board[r][c] != ".":
            return _backtrack(new_r, new_c)

        box_idx = (r // 3) * 3 + (c // 3)
        for num in range(1, 10):
            if num not in filledRows[r] and num not in filledCols[c] and num not in filledBoxes[box_idx]:
                board[r][c] = str(num)
                filledRows[r].add(num)
                filledCols[c].add(num)
                filledBoxes[box_idx].add(num)

                if _backtrack(new_r, new_c):
                    return True

                board[r][c] = "."
                filledRows[r].remove(num)
                filledCols[c].remove(num)
                filledBoxes[box_idx].remove(num)
        return False

    _backtrack(0, 0)


def solve_sudoku_2(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    empties = []

    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                empties.append((i,j))
            else:
                val = int(board[i][j])
                rows[i].add(val)
                cols[j].add(val)
                boxes[(i//3)*3 + (j//3)].add(val)

    def candidates(r, c):
        return {1,2,3,4,5,6,7,8,9} - rows[r] - cols[c] - boxes[(r//3)*3 + (c//3)]

    def backtrack():
        if not empties:
            return True

        # Pick the empty cell with fewest candidates
        empties.sort(key=lambda x: len(candidates(x[0], x[1])))
        r, c = empties.pop(0)

        for num in candidates(r, c):
            board[r][c] = str(num)
            rows[r].add(num)
            cols[c].add(num)
            boxes[(r//3)*3 + (c//3)].add(num)

            if backtrack():
                return True

            board[r][c] = '.'
            rows[r].remove(num)
            cols[c].remove(num)
            boxes[(r//3)*3 + (c//3)].remove(num)

        empties.insert(0, (r,c))
        return False

    backtrack()

def display_board(board):
    for i in range(9):
        for j in range(9):
            if j == 2 or j == 5:
                print(board[i][j] + " | ", end="")
            else:
                print(board[i][j] + " ", end="")
        if i == 2 or i == 5:
            print("\n------+-------+------", end="")
        print()


if __name__ == "__main__":
    board = [[".",".","6",".",".","5",".",".","."],
             [".",".","8",".","9",".",".",".","."],
             [".",".","2",".",".",".","8","1","7"],
             ["4",".",".","3",".","8",".",".","."],
             [".","3",".",".","5",".",".","4","."],
             [".",".",".","2",".","6",".",".","9"],
             ["3","1","5",".",".",".","4",".","."],
             [".",".",".",".","7",".","9",".","."],
             [".",".",".","1",".",".","6",".","."]]

    display_board(board)
    if is_valid_sudoku(board):
        print("\n\nBoard is valid, solving...\n")
    else:
        print("\n\nBoard is invalid, cannot solve.\n")
        exit(1)
    solve_sudoku(board)
    display_board(board)


'''
The following board is unsolvable. The board is valid, but there are no solutions. The solver will attempt to solve it, but will fail and return the original board.
Unsolved board: 
. . 6 | . . 5 | . . . 
. . 8 | . 9 . | . . . 
. . 2 | . . . | 8 1 7 
------+-------+------ 
4 . . | 3 . 8 | . . . 
. 3 . | . 5 . | . 4 . 
. . . | 2 . 6 | . . 9 
------+-------+------ 
3 1 5 | . . . | 4 . . 
. . . | . 7 . | 9 . . 
. . . | 1 . . | 6 . .

'''