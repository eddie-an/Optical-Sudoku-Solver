import { useState } from 'react'

import './App.css'
import SudokuGrid from './SudokuGrid';

function App() {
  const [image, setImage] = useState(null); // Binary image input
  const [previewUrl, setPreviewUrl] = useState(null); // Used for img src attribute

  const [isSolved, setIsSolved] = useState(false); // Keeps track of whether the selected image is submitted to be solved
  const [isLoading, setIsLoading] = useState(false); // Computer vision API takes a while to solve sometimes. Prevents solve button spam

  const [board, setBoard] = useState([]); // String matrix consisting of the solved board
  const [givenNumbers, setGivenNumbers] = useState([]); // Boolean matrix that shows if a cell was given in the image or not
  const [isBoardValid, setIsBoardValid] = useState(false); // Ensures the returne board is valid

  function checkBoard(board) {
    board.forEach((row)=> {
      row.forEach((cell)=> {
        if (cell === '.') {
          return false;
        }
      })
    })
    return true;
  }
  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file == null) return;
    setImage(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleSolve = async (image) => {
    setIsSolved(false);
    if (!image) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    const url = "http://localhost:8000/solve"
    try {
      const response = await fetch(
        url, 
        {
          method: "POST",
          body: formData
        }
      );
      if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
      }
      const result = await response.json();
      setBoard(result.board);
      setGivenNumbers(result.given);
      setIsBoardValid(checkBoard(board));
    }
    catch (error) {
      console.error(error.message);
      setIsBoardValid(false);
    } finally {
      setIsSolved(true);
      setIsLoading(false);
    }

  };

  return (
    <div className='main-container'>
      <nav className='navigation-bar'>
        <h1>Sudoku Solver</h1>
      </nav>
      <div className="upload-card">
        <label htmlFor="choose-file-input" className="choose-file-label">
          Choose Sudoku Image
        </label>
        <input type='file'
              accept='image/*'
              onChange={handleImageChange}
              id="choose-file-input">
        </input>
        {
          previewUrl && 
          <img src={previewUrl} alt="preview" className="preview-image"/>
        }
      </div>
      { image && <button id='solve-button' disabled={isLoading} onClick={()=> handleSolve(image)}>{isLoading ? "Solving..." : "Solve"}</button>}
      {isSolved && (
        isBoardValid
          ? <SudokuGrid board={board} givenCells={givenNumbers} />
          : <div className="error-message">
              Couldn't read the puzzle. Try another photo with better lighting
              and make sure all 4 corners of the grid are visible.
            </div>
      )}

    </div>
  )
}

export default App
