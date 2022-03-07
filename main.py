import torch
from torch.nn import CrossEntropyLoss
import pandas as pd
from train_data_loader import TrainFFTDataset
from test_data_loader import TestFFTDataset
from network import DHRN
from utils import *


def train(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset_train = TrainFFTDataset(opts)
    dataset_test = TestFFTDataset(opts)

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.train_batch_shuffle, 
                                                    num_workers = opts.n_threads)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.test_batch_shffle, 
                                                    num_workers = opts.n_threads)

    # get the model
    model = DHRN().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = opts.learning_rate, betas=(0.9,0.99))     

    train_detection_loss_list = []
    train_recognition_loss_list = []
    train_detection_acc_list = []
    train_recognition_acc_list = []

    test_detection_loss_list = []
    test_recognition_loss_list = []
    test_detection_acc_list = []
    test_recognition_acc_list = []

    for epoch in range(opts.epochs):

        train_batch_num = 0
        train_detection_loss = 0.0
        train_recognition_loss = 0.0

        model.train()
        counts_detection = 0
        counts_recognition = 0

        for seq, label_detection, label_recogition in data_loader_train:
            # cpu--->gpu
            seq = seq.to(device)
            label_detection = label_detection.to(device)
            label_recogition = label_recogition.to(device)
            seq = seq.unsqueeze(1)

            optimizer.zero_grad()
            # obtain prediction results
            detection_pred, _ = model(seq)
            _, recognition_pred = model(seq)

            # calculate loss 
            loss_detection = CrossEntropyLoss(detection_pred, label_detection.view(-1))
            loss_recognition = CrossEntropyLoss(recognition_pred, label_recogition.view(-1))

            # backward loss
            loss_detection.backward()
            loss_recognition.backward()
            optimizer.step()

            train_batch_num += 1
            train_detection_loss += loss_detection.item()
            train_recognition_loss += loss_recognition.item()

            pred_detection = detection_pred.argmax(dim = 1, keepdims = True)
            pred_recognition = recognition_pred.argmax(dim = 1, keepdims = True)

            counts_detection += pred_detection.cpu().eq(label_detection.cpu().view_as(pred_detection)).sum().item()
            counts_recognition += pred_recognition.cpu().eq(label_recogition.cpu().view_as(pred_recognition)).sum().item()

        # calculate accuracy
        detection_acc = counts_detection * 1.0 / len(data_loader_train.dataset)
        recognition_acc = counts_recognition * 1.0 / len(data_loader_train.dataset)

        # add accuracy and loss into list
        train_detection_loss_list.append(train_detection_loss / len(data_loader_train.dataset))
        train_recognition_loss_list.append(train_recognition_loss / len(data_loader_train.dataset))
        train_detection_acc_list.append(detection_acc)
        train_recognition_acc_list.append(recognition_acc)

        # list ---> dataframe
        train_detection_loss_dataframe = pd.DataFrame(data = train_detection_loss_list)
        train_recognition_loss_dataframe = pd.DataFrame(data = train_recognition_loss_list)
        train_detection_acc_dataframe = pd.DataFrame(data = train_detection_acc_list)
        train_recognition_acc_dataframe = pd.DataFrame(data = train_recognition_acc_list)

        # write 'csv' file
        train_detection_loss_dataframe.to_csv("./outputs/train_detection_loss.csv", index = False)
        train_recognition_loss_dataframe.to_csv("./outputs/train_recognition_loss.csv", index = False)
        train_detection_acc_dataframe.to_csv("./outputs/train_detection_acc.csv", index = False)
        train_recognition_acc_dataframe.to_csv("./outputs/train_recognition_acc.csv", index = False)

        # model eval
        model.eval()

        test_detection_real = []
        test_recognition_real = []
        test_detection_prediction = []
        test_recognition_prediction = []

        counts_detection_test = 0
        counts_recognition_test = 0
        test_loss_detection = 0
        test_loss_recognition = 0
        test_batch_num = 0

        outs_detection = []
        outs_recognition = []
        labels_detection = []
        labels_recognition = []
        with torch.no_grad():
            for test_seq, test_label_detection, test_label_recogition in data_loader_test:
                # cpu--->gpu
                test_seq = test_seq.to(device)
                test_label_detection = test_label_detection.to(device)
                test_label_recogition = test_label_recogition.to(device)
                test_seq = test_seq.unsqueeze(1)
                
                # obtain prediction results
                test_detection_pred = model(test_seq)
                test_recognition_pred = model(test_seq)

                outs_detection.append(test_detection_pred.cpu())
                outs_recognition.append(test_recognition_pred.cpu())

                labels_detection.append(test_label_detection.cpu())
                labels_recognition.append(test_label_recogition.cpu())

                # calculate loss 
                loss_detection = CrossEntropyLoss(test_detection_pred, test_label_detection.view(-1))
                loss_recognition = CrossEntropyLoss(test_recognition_pred, test_label_recogition.view(-1))

                test_loss_detection += loss_detection.item()
                test_loss_recognition += loss_recognition.item()
                test_batch_num += 1

                test_detection_real += list(test_label_detection.data.cpu().numpy().flatten())
                test_recognition_real += list(test_label_recogition.data.cpu().numpy().flatten())

                test_detection_prediction += list(test_detection_pred.data.cpu().numpy().flatten())
                test_recognition_prediction += list(test_recognition_pred.data.cpu().numpy().flatten())

                test_pred_detection = test_detection_pred.argmax(dim = 1, keepdims = True)
                test_pred_recognition = test_recognition_pred.argmax(dim = 1, keepdims = True)

                counts_detection_test += test_pred_detection.cpu().eq(test_label_detection.cpu().view_as(test_pred_recognition)).sum().item()
                counts_recognition_test += test_pred_recognition.cpu().eq(test_label_recogition.cpu().view_as(test_pred_recognition)).sum().item()


        outs_detection = torch.cat(outs_detection, dim=0)
        outs_recognition = torch.cat(outs_recognition, dim=0)

        labels_detection = torch.cat(labels_detection).reshape(-1)
        labels_recognition = torch.cat(labels_recognition).reshape(-1)

        # calculate accuracy
        avg_detection_acc = counts_detection_test * 1.0 / len(data_loader_test.dataset)
        avg_recognition_acc = counts_recognition_test * 1.0 / len(data_loader_test.dataset)

        # add accuracy and loss into list
        test_detection_acc_list.append(avg_detection_acc)
        test_recognition_acc_list.append(avg_recognition_acc)

        test_detection_loss_list.append(test_loss_detection / len(data_loader_test.dataset))
        test_recognition_loss_list.append(test_loss_recognition / len(data_loader_test.dataset))
        
        # print related information
        print("epoch: %d, train_detection_loss: %.5f, train_recognition_loss: %.5f, \
                test_detection_loss: %.5f, test_recognition_loss: %.5f, \
                test_detection_acc: %.5f, test_recognition_acc: %.5f" % 
                (epoch, train_detection_loss / train_batch_num, train_recognition_loss / train_batch_num, 
                test_loss_detection / test_batch_num, test_loss_recognition / test_batch_num, 
                avg_detection_acc,avg_recognition_acc))

        # list ---> dataframe
        test_detection_loss_datadrame = pd.DataFrame(data = test_detection_loss_list)
        test_recognition_loss_dataframe = pd.DataFrame(data = test_recognition_loss_list)
        test_detection_acc_dataframe = pd.DataFrame(data = test_detection_acc_list)
        test_recognition_acc_dataframe = pd.DataFrame(data = test_recognition_acc_list)

        # write 'csv' file
        test_detection_loss_datadrame.to_csv("./outputs/test_detection_loss.csv", index = False)
        test_recognition_loss_dataframe.to_csv("./outputs/test_recognition_loss.csv", index = False)
        test_detection_acc_dataframe.to_csv("./outputs/test_detection_acc.csv", index = False)
        test_recognition_acc_dataframe.to_csv("./outputs/test_recognition_acc.csv", index = False)

        # Filter and save models 
        if avg_detection_acc > 0.90:
            torch.save(model.state_dict(), "./models/model_cavitation_{epoch}_{value}.pth".format(epoch = epoch, value = avg_detection_acc))
        if avg_recognition_acc > 0.90:
            torch.save(model.state_dict(), "./models/model_recognition_{epoch}_{value}.pth".format(epoch = epoch, value = avg_recognition_acc))
    # plot curves
    draw_train_detection(train_detection_loss_list, train_detection_acc_list)
    draw_train_recognition(train_recognition_loss_list, train_recognition_acc_list)
    draw_test_detection(test_detection_loss_list, test_detection_acc_list)
    draw_test_recognition(test_recognition_loss_list, test_recognition_acc_list)
    draw_roc_confusion_detection(outs_detection, labels_detection)
    draw_roc_confusion_recognition(outs_recognition, labels_recognition)


