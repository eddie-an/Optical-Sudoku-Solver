# Optical-Sudoku-Solver

This app takes an input sudoku image and solves it using computer vision algorithms and machine learning models. The repository is split into 3 folders:
- computer-vision-api
- report
- webpage

The `computer-vision-api` folder contains the code for the computer vision algorithm. The `report` folder contains an informal paper describing the algorithm, written in LaTeX. The `webpage` folder contains code for the front end of the application. This is where the API is called.

For more information, check the `README.md` files in respective folders.

## Running the app

To run this app locally, you need the computer vision API running on the local server using Uvicorn and the front end page running with NPM.

Refer to the `computer-vision-api/README.md` to run the API. 

To run the front end, run `npm install` then `npm run dev`.