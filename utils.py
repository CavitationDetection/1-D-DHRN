import os
import torch
import itertools
import numpy as np
from itertools import cycle
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix


def create_dirs(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

##  plot train loss and acc curves
def draw_train_detection(train_detection_loss_list, train_detection_acc_list):
    # plot train detection loss
    plt.plot(list(range(len(train_detection_loss_list))), train_detection_loss_list, 'o-')
    plt.title('Train Detection Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("./figs/train_detection_loss.jpg")
    plt.close()

    # plot train detection acc 
    plt.plot(list(range(len(train_detection_acc_list))), train_detection_acc_list, '.-')
    plt.title('Train Detection Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim([0.0, 1.05])
    plt.savefig('./figs/train_detection_acc.jpg')
    plt.close()

def draw_train_recognition(train_recognition_loss_list, train_recognition_acc_list):
    # plot train recognition loss
    plt.plot(list(range(len(train_recognition_loss_list))), train_recognition_loss_list, 'o-')
    plt.title('Train Recognition Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("./figs/train_recognition_loss.jpg")
    plt.close()

    # plot train recognition acc 
    plt.plot(list(range(len(train_recognition_acc_list))), train_recognition_acc_list, '.-')
    plt.title('Train Recognition Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim([0.0, 1.05])
    plt.savefig('./figs/train_recognition_acc.jpg')
    plt.close()


## plot test loss and acc curves
def draw_test_detection(test_detection_loss_list, test_detection_acc_list):
    # plot test detection loss
    plt.plot(list(range(len(test_detection_loss_list))), test_detection_loss_list, 'o-')
    plt.title('Test Detection Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./figs/test_detection_loss.jpg')
    plt.close()

    # plot test detection acc
    plt.plot(list(range(len(test_detection_acc_list))), test_detection_acc_list, '.-')
    plt.title('Test Detection Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim([0.0, 1.05])
    plt.savefig('./figs/test_recognition_acc.jpg')
    plt.close()

def draw_test_recognition(test_recognition_loss_list, test_recognition_acc_list):
    # plot test recognition loss
    plt.plot(list(range(len(test_recognition_loss_list))), test_recognition_loss_list, 'o-')
    plt.title('Test Recognition Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./figs/test_recognition_loss.jpg')
    plt.close()

    # plot test acc
    plt.plot(list(range(len(test_recognition_acc_list))), test_recognition_acc_list, '.-')
    plt.title('Test Recognition Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim([0.0, 1.05])
    plt.savefig('./figs/test_recognition_acc.jpg')
    plt.close()

## plot detection ROC and confusion matrix 
def draw_roc_confusion_detection(outs_detection, labels_detection):
    pre = outs_detection.detach().numpy()
    classes = pre.shape[1]
    y_test = torch.nn.functional.one_hot(labels_detection,classes).numpy()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lw = 2
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]), 
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darkgreen'])
    for i, color in zip(range(classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./figs/roc_detection.jpg')
    plt.close()
    with open("./outputs/roc_detection.csv",'w') as f:
        [f.write('{0},{1}\n'.format(key,value)) for key,value in fpr.items()]

    predictions = pre.argmax(axis=-1)
    truelabel = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)

    # precision recall f1
    precision_micro = precision_score(truelabel,predictions,average='micro')
    recall_micro = recall_score(truelabel,predictions,average='micro')
    f1score_micro = f1_score(truelabel,predictions,average='micro')

    precision_macro = precision_score(truelabel,predictions,average='macro')
    recall_macro = recall_score(truelabel,predictions,average='macro')
    f1score_macro = f1_score(truelabel,predictions,average='macro')
    accuracyscore = accuracy_score(truelabel,predictions)
    plt.figure()

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix')
    plt.colorbar()
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
        horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=30)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./figs/confusionmatrix_detection.jpg', dpi=350)
    plt.close()

    classificationreport = classification_report(truelabel,predictions)
    print('-----------------Cavitation Dtection-------------------')
    print("Classification Report:\n{}".format(classificationreport))
    print("Confusion Matrix:\n{}".format(cm))
    print("Accuracy:{}".format(accuracyscore))
    print('Precision(micro):{}'.format(precision_micro))
    print('Precision(macro):{}'.format(precision_macro))
    print("Recall(micro):{}".format(recall_micro))
    print("Recall(macro):{}".format(recall_macro))
    print("F1_score(micro):{}".format(f1score_micro))
    print("F1_score(macro):{}".format(f1score_macro))

## plot recognition ROC and confusion matrix 
def draw_roc_confusion_recognition(outs_recognition, labels_recognition):
    pre = outs_recognition.detach().numpy()
    classes = pre.shape[1]
    y_test = torch.nn.functional.one_hot(labels_recognition,classes).numpy()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lw = 2
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]), 
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darkgreen'])
    for i, color in zip(range(classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./figs/roc_recognition.jpg')
    plt.close()
    with open("./outputs/roc_recognition.csv",'w') as f:
        [f.write('{0},{1}\n'.format(key,value)) for key,value in fpr.items()]

    predictions = pre.argmax(axis=-1)
    truelabel = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)

    # precision recall f1
    precision_micro = precision_score(truelabel,predictions,average='micro')
    recall_micro = recall_score(truelabel,predictions,average='micro')
    f1score_micro = f1_score(truelabel,predictions,average='micro')

    precision_macro = precision_score(truelabel,predictions,average='macro')
    recall_macro = recall_score(truelabel,predictions,average='macro')
    f1score_macro = f1_score(truelabel,predictions,average='macro')
    accuracyscore = accuracy_score(truelabel,predictions)
    plt.figure()

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix')
    plt.colorbar()
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=30)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./figs/confusionmatrix_recognition.jpg', dpi=350)
    plt.close()

    classificationreport = classification_report(truelabel,predictions)
    print('-----------------Cavitation Intensity Recognition-------------------')
    print("Classification Report:\n{}".format(classificationreport))
    print("Confusion Matrix:\n{}".format(cm))
    print("Accuracy:{}".format(accuracyscore))
    print('Precision(micro):{}'.format(precision_micro))
    print('Precision(macro):{}'.format(precision_macro))
    print("Recall(micro):{}".format(recall_micro))
    print("Recall(macro):{}".format(recall_macro))
    print("F1_score(micro):{}".format(f1score_micro))
    print("F1_score(macro):{}".format(f1score_macro))