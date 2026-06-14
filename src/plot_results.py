import matplotlib.pyplot as plt


def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend([
        "Train",
        "Validation"
    ])

    plt.savefig(
        "outputs/accuracy_plot.png"
    )

    plt.show()


def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend([
        "Train",
        "Validation"
    ])

    plt.savefig(
        "outputs/loss_plot.png"
    )

    plt.show()