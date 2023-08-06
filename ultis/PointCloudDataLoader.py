import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import ultis.provider as provider

class PointCloudDataLoader(Dataset):
    def __init__(self, root, split, num_pcs = 196, startIndex_split = None, endIndex_split = None):
        self.root = root
        
        assert split in ["train", "test"]
        x_data = np.load(os.path.join(root, f"X_{split}.npy"))[startIndex_split:endIndex_split]
        x_data = x_data.reshape(x_data.shape[0], 196, 5)
        x_data = np.transpose(x_data, (0, 2, 1))
        x_data = np.insert(x_data, 5, 0, axis = 1)
        x_data = x_data.astype(np.float32)
        print(x_data.shape)
        self.x_data = x_data
        self.y_data = np.load(os.path.join(root, f"y_{split}.npy"))[startIndex_split:endIndex_split]
        self.len = self.x_data.shape[0]
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len