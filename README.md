# Optical-Sudoku-Solver
Solves Sudoku puzzles using computer vision


## Dataset
[Sudoku Dataset](https://github.com/wichtounet/sudoku_dataset)

If we do the ML approach for digit recognition, this may be good to use:
[MNIST handwritten digits](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)


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