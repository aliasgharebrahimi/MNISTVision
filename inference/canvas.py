import cv2
import numpy as np


canvas = np.ones((400, 400), dtype=np.uint8) * 255

drawing = False


def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("MNISTVision")
cv2.setMouseCallback("MNISTVision", draw)


while True:
    cv2.imshow("MNISTVision", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") or key == ord("C"):
        canvas[:] = 255

    elif key == 27:
        break


cv2.destroyAllWindows()