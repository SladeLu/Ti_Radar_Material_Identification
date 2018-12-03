from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
BATCH_SIZE = 64
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):

        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        single_signal_label = self.labels[index]
        sig_as_np = np.asarray(self.data.iloc[index][1:]).reshape(self.height,self.width)
        if self.transforms is not None:
            sig_as_tensor = self.transforms(sig_as_sig)
        return (sig_as_tensor, single_signal_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    train_data = CustomDatasetFromCSV('./data/formal_test.csv',128,39, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # print(train_data.train_data.size())     # (60000, 28, 28)
    # print(train_data.train_labels.size())   # (60000)
    # print(train_data)