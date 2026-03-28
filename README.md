# Optical-Sudoku-Solver
Solves Sudoku puzzles using computer vision


## Dataset
[Sudoku Dataset](https://github.com/wichtounet/sudoku_dataset)

### Digit Recognition Dataset
We use the [Chars74K EnglishFnt](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) subset — computer-rendered digits across 1,016 fonts, well-matched to printed Sudoku puzzles.

**No manual download required.** Run `DigitRecognition.ipynb` and the first cell will automatically download and extract the dataset into `data/`.


## Usage

### To train the digit recognition model
Run `DigitRecognition.ipynb` — this downloads the required datasets, trains the SVM, and saves the model to `models/digit_model.pkl`.

> **Note:** `models/digit_model.pkl` is not included in the repository (file size exceeds GitHub's 100MB limit). You must generate it by running `DigitRecognition.ipynb` before using `Image_Processing.ipynb`.

### To evaluate digit recognition performance
Run `Evaluation.ipynb` after `DigitRecognition.ipynb`. Requires `models/digit_model.pkl` and `data/test_puzzles.txt` to exist.

## Installing dependencies
Create environment: `python -m venv venv`

Activate it:
- Windows: `.\venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

Install your requirements: `pip install -r requirements.txt`

To deactivate virtual environment: `deactivate`

## Running Jupyter Notebook and Linking Venv to Jupyter
We will be using Jupyter notebook to develop algorithms at the start. Near the deadline, we will migrate everything to python files instead. Jupyter notebook is best for image processing since you can see the intermediate steps.

After installing dependencies, you must register the virtual environment as a Jupyter kernel:
1. `python -m ipykernel install --user --name=sudoku-env --display-name "Python (Sudoku Project)"`
2. Run: `jupyter notebook`
3. In your notebook, go to **Kernel -> Change Kernel** and select **"Python (Sudoku Project)"**.