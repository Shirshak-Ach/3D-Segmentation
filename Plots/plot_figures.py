import matplotlib.pyplot as plt

def plot_loss(num_epochs, overall_train_loss_per_epoch, overall_test_loss_per_epoch):
    plt.plot(range(1, num_epochs + 1),overall_train_loss_per_epoch,label='Training Loss')
    plt.plot(range(1, num_epochs + 1),overall_test_loss_per_epoch,label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()
    plt.show()

def plot_IOU_Jaccard(num_epochs, overall_train_jaccard_per_epoch, overall_test_jaccard_per_epoch):
    plt.plot(range(1, num_epochs + 1),overall_train_jaccard_per_epoch,label='Training IOU')
    plt.plot(range(1, num_epochs + 1),overall_test_jaccard_per_epoch,label='Testing IOU')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.title('IOU/Jaccard Score Over Epochs')
    plt.legend()
    plt.show()
    plt.show()

def plot_Accuracy(num_epochs, overall_train_acc_per_epoch, overall_test_acc_per_epoch):
    plt.plot(range(1, num_epochs + 1),overall_train_acc_per_epoch,label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1),overall_test_acc_per_epoch,label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Score Over Epochs')
    plt.legend()
    plt.show()
    plt.show()
