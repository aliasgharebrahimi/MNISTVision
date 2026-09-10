import torch
import matplotlib.pyplot as plt

def confusion_matrix(pred, labels, num_classes):
    pred = torch.argmax(pred, dim=1)

    cm = torch.zeros(num_classes, num_classes)

    for true, predicted in zip(labels, pred):
        cm[true, predicted] += 1

    return cm

def plot_confusion_matrix(cm):

    cm = cm.cpu()

    fig, ax = plt.subplots(figsize=(8, 7))

    # رسم ماتریس
    im = ax.imshow(
        cm,
        cmap="Greens",
        interpolation="nearest"
    )

    # عنوان و نام محورها
    ax.set_title(
        "Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12
    )

    # شماره کلاس‌ها
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # نوار رنگ
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        "Number of Samples",
        rotation=-90,
        va="bottom"
    )

    # نوشتن عدد داخل هر خانه
    threshold = cm.max() / 2

    for i in range(10):
        for j in range(10):

            value = cm[i, j].item()

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=10
            )

    # مرتب‌سازی ظاهر
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10
    )

    plt.tight_layout()
    plt.show()