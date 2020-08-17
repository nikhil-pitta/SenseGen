import numpy as np
import os
import sys
from scipy.io import loadmat

TRAIN_DATA_FILE = "dataset/train/Inertial Signals/total_acc_x_train.txt"

ECG_data_dir = 'dataset/ecg_data/1 NSR'

def load_training_data(dataset='acc'):
    """ Returns a matrix of training data.
    shape of result = (n_exp, len)
    """
    if dataset=='acc':
        data = np.loadtxt(TRAIN_DATA_FILE)
        return data.T
    elif dataset == 'ecg':
        data_files = [f for f in os.listdir(ECG_data_dir)
                        if f.endswith('.mat')]
        data_list = []
        for fname in data_files:
            exp_data =  loadmat(
                        os.path.join(ECG_data_dir, fname))['val'].astype(np.float32).ravel()
            exp_data = (exp_data - np.mean(exp_data)) / np.std(exp_data) # normalize for numerical stability
            exp_data = exp_data[::2] # subsample to reduce computational cost
            data_list.append(exp_data)
        data = np.stack(data_list)
        return data
          

class DataLoader(object):
    def __init__(self, data, batch_size=128, num_steps=1):
        self.batch_size = batch_size
        self.n_data, self.seq_len = data.shape
        #num_batches = (self.n_data // self.batch_size) * self.batch_size
        #TODO(malzantot): needs to be improved to utilize all data
        self._data = data[:self.batch_size , :]
        
        self.num_steps = num_steps
        self._data = self._data.reshape((self.batch_size, self.seq_len, 1))
        self._reset_pointer()

    def _reset_pointer(self):
            self.pointer = 0 

    def reset(self):
        self._reset_pointer()

    def has_next(self):
        return self.pointer + self.num_steps < self.seq_len - 1

    def next_batch(self):
       batch_xs = self._data[:, self.pointer:self.pointer+self.num_steps, :]
       batch_ys = self._data[:, self.pointer+1:self.pointer+self.num_steps+1, :]
       self.pointer = self.pointer + self.num_steps
       return batch_xs, batch_ys
        

