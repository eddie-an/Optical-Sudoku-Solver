def isValidSudoku(board) -> bool:
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

def solveSudoku(board) -> None:
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

    def backtrack(r: int, c: int):
        if r == 9:
            return True
        
        new_r = (r+1) if c == 8 else r
        new_c = (c+1) % 9
        if (board[r][c] != "."):
            return backtrack(new_r, new_c)

        sectionNumber = (r // 3) * 3 + (c // 3)
        for num in range(1, 10):

            if (num not in filledRows[r]) and (num not in filledCols[c]) and (num not in filledBoxes[sectionNumber]):
                board[r][c] = str(num)
                filledRows[r].add(num)
                filledCols[c].add(num)
                filledBoxes[sectionNumber].add(num)
                if (backtrack(new_r, new_c)):
                    return True
        
                board[r][c] = "."
                filledRows[r].remove(num)
                filledCols[c].remove(num)
                filledBoxes[sectionNumber].remove(num)
        return False

    solved = False
    backtrack(0,0)

def displayBoard(board):
    for row in board:
        print(row)
board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
displayBoard(board)
solveSudoku(board)
displayBoard(board)