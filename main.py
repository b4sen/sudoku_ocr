from extractor import find_puzzle
from extractor import extract_digit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from solver import Solver

import numpy as np
import imutils
import cv2

model = load_model('model/digit_rec.h5')

def generate_board(img: str):
    img = cv2.imread(img)
    img = imutils.resize(img, width=600)

    (puzzleImage, warped) = find_puzzle(img)

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    cellLocs = []
    board = np.zeros((9,9), dtype='int')
    for y in range(9):
        row = []
        for x in range(9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            row.append((startX, startY, endX, endY))

            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)

            if digit is not None:
                roi = cv2.resize(digit, (28,28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                pred = model.predict(roi).argmax(axis=1)[0]
                board[y][x] = pred
        cellLocs.append(row)
    return cellLocs, board, puzzleImage

cellLocs, board, puzzleImg = generate_board('test.jpeg')
solver = Solver()
solver.set_board(board)
solver.solve()



for (cellRow, boardRow) in zip(cellLocs, solver.board):
    for (box, digit) in zip(cellRow, boardRow):
        startX, startY, endX, endY = box
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY

        cv2.putText(puzzleImg, str(digit), (textX, textY),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Sudoku Result", puzzleImg)
cv2.waitKey(0)
