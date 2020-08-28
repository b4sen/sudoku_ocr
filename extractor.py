import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

IMG = cv2.imread('test.jpeg')

def find_puzzle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            puzzleCnt = approx
            break
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. Try debugging your thresholding and contour steps."))
    
    puzzle = four_point_transform(img, puzzleCnt.reshape(4,2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4,2))

    return (puzzle, warped)

def extract_digit(cell, debug=False):

    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None
    
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    # if less than 3% of the mas is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
        
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    if debug:
        cv2.imshow('digit', digit)
        cv2.waitKey(0)
    
    return digit




