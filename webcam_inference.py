import cv2
import torch
from torchvision import transforms

from neural_network import model


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# آماده سازی مدل
model.to(device)
model.eval()


# تبدیل تصویر برای MNIST
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# باز کردن وبکم
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # کپی برای نمایش
    display = frame.copy()


    # تبدیل به خاکستری
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # گرفتن بخش وسط تصویر
    h, w = gray.shape

    x1 = w // 2 - 100
    y1 = h // 2 - 100

    x2 = w // 2 + 100
    y2 = h // 2 + 100


    roi = gray[y1:y2, x1:x2]


    # شبیه سازی MNIST:
    # عدد سفید روی زمینه سیاه
    _, roi = cv2.threshold(
        roi,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )


    # آماده کردن برای مدل
    img = transform(roi)

    # اضافه کردن batch dimension
    img = img.unsqueeze(0)

    img = img.to(device)


    # پیش بینی
    with torch.no_grad():

        output = model(img)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    # کشیدن مربع محل نوشتن عدد
    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # نوشتن نتیجه
    cv2.putText(
        display,
        f"Number: {prediction}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )


    # نمایش
    cv2.imshow(
        "MNISTVision Webcam",
        display
    )


    # خروج با q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()