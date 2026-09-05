import cv2
import numpy as np

from inference.preprocessing import preprocess
from inference.predictor import predict


canvas = np.ones((400, 400), dtype=np.uint8) * 255

drawing = False
prediction = None


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
    display_canvas = canvas.copy()

    if prediction is not None:
        cv2.putText(
            display_canvas,
            f"Prediction: {prediction}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            0,
            2
        )

    cv2.imshow("MNISTVision", display_canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") or key == ord("C"):
        canvas[:] = 255
        prediction = None

    elif key == ord("p") or key == ord("P"):
        image = preprocess(canvas)
        prediction = predict(image)
        print(f"Prediction: {prediction}")

    elif key == 27:
        break


cv2.destroyAllWindows()