## Sudoku Solver with OCR

In this little project I trained a CNN with Tensorflow on the MNIST dataset to be able to recognize digits.
The recognized digits will be arranged into a sudoku board and a backtracking algorithm tries to solve it.
Once a solution is found a picture of the solved sudoku is displayed.

## Usage

You can find the CNN in the `model.py`. Using the `train.py` with `-dl` or `--download` arguments will download the official MNIST dataset and save it in binary form into the `data` folder. If you already have them downloaded, you can skip the arguments.

The `extractor.py` has the methods to extract the sudoku board and the digits in the cells.
The `solver.py` contains the backtracking algorithm.
The `main.py` is the entrypoint of this tool. The models and the input images are hardcoded.
