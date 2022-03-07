import pandas as pd
import torch
import os

class TrainFFTDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.train_root
        self.csv_files = [file for file in list(sorted(os.listdir(self.root))) if '.csv' in file] 
        self.labeldict = self.labelDict()

    def labelDict(self):
        label_df = pd.read_csv(self.opt.train_label_path)
        labels = label_df['label'].values
        index2label = {}
        for i, label in enumerate(labels):
            index2label[i] = label
        return index2label

    def getFileIndex(self, csv_file):
        return int(csv_file.strip().replace('.csv', ''))

    def trunate_and_pad(self, data, pad=0):
        "数据截断或填充"
        if len(data) >= self.opt.mess_length:
            data = data[:self.opt.mess_length]
        else:
            padding = [pad] * (self.opt.mess_length - len(data))
            data += padding
        assert len(data) == self.opt.mess_length
        return data

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.csv_files[idx])
        infodata = pd.read_csv(file_path, header=None)
        infodata = infodata[0].tolist()
        infodata = self.trunate_and_pad(infodata)
        infodata = torch.as_tensor(infodata, dtype=torch.float32)
        labelIndex = self.getFileIndex(self.csv_files[idx])
        label_detection = self.labeldict[labelIndex]
        label_recognition = self.labeldict[labelIndex]

        if str(label_detection) == "0":
            label_detection = [0]
        elif str(label_detection) == "1":
            label_detection = [1]
        elif str(label_detection) == "2":
            label_detection = [1]
        elif str(label_detection) == "3":
            label_detection = [1]

        if str(label_recognition) == "0":
            label_recognition = [0]
        elif str(label_recognition) == "1":
            label_recognition = [1]
        elif str(label_recognition) == "2":
            label_recognition = [2]
        elif str(label_recognition) == "3":
            label_recognition = [3]

        label_detection = torch.as_tensor(label_detection, dtype=torch.int64)
        label_recognition = torch.as_tensor(label_recognition, dtype=torch.int64)
        return infodata, label_detection,label_recognition

    def __len__(self):
        return len(self.csv_files)




        
