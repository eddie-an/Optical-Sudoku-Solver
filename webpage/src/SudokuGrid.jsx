import { useState } from 'react'

function SudokuGrid({board, givenCells}) {
    if (!board || !givenCells) return null;
    if (board.length != givenCells.length || board.length != 9 || board[0].length != givenCells[0].length || board[0].length != 9) {
        return null;
    }

    return (
        <div className='sudoku-grid'>
            {board.map((row, rowIndex)=> (
                <div key={rowIndex} className='sudoku-row'>
                    {row.map((cellVal, colIndex)=> {
                        let classes = ["sudoku-cell"];
                        let cellClass = "";
                        let thickVerticalBorder = "";
                        let thickHorizontalBorder = "";
                        if (givenCells[rowIndex][colIndex] == true) classes.push("given-cell");
                        if (colIndex % 3 === 0) classes.push("thick-left-border-cell");
                        else if (colIndex % 3 === 2) classes.push("thick-right-border-cell");

                        if (rowIndex % 3 === 0) classes.push("thick-top-border-cell");
                        else if (rowIndex % 3 === 2) classes.push("thick-bottom-border-cell");
                        return (
                            <div key={`${rowIndex}-${colIndex}`} className={classes.join(' ')}>
                                <p>{cellVal}</p>
                            </div>
                        );
                    })}
                </div>
            ))}
        </div>
    );
}

export default SudokuGrid;