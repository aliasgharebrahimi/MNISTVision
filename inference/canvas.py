import cv2
import numpy as np

from inference.preprocessing import preprocess
from inference.predictor import predict


canvas = np.ones((400, 400), dtype=np.uint8) * 255

drawing = False
prediction = None


def draw(event, x, y, flags, param):
    global drawing, prediction

    if event == cv2.EVENT_LBUTTONDOWN:

        button = handle_button(x, y)

        if button == "predict":
            image = preprocess(canvas)
            prediction = predict(image)
            print(f"Prediction: {prediction}")
            return

        elif button == "clear":
            canvas[:] = 255
            prediction = None
            return

        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def handle_button(x, y):
    # Predict button
    if 250 <= x <= 390 and 350 <= y <= 390:
        return "predict"

    # Clear button
    if 100 <= x <= 240 and 350 <= y <= 390:
        return "clear"

    return None

def draw_buttons(image):
    cv2.rectangle(image, (100, 350), (240, 390), 0, -1)
    cv2.rectangle(image, (250, 350), (390, 390), 0, -1)

    cv2.putText(
        image,
        "Clear",
        (135, 377),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )

    cv2.putText(
        image,
        "Predict",
        (275, 377),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )
cv2.namedWindow("MNISTVision")
cv2.setMouseCallback("MNISTVision", draw)


while True:
    display_canvas = canvas.copy()
    draw_buttons(display_canvas)

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