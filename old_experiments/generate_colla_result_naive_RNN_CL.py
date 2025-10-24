#%%
## Load my files ##
import sys
sys.path.append('..')
from source.utils import get_sequence

## Load standard files ##
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch import from_numpy as tnsr
from scipy.stats import bernoulli
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as dist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
import argparse 
import pickle
#%%
n_community = 2
n_members = 3

tokens = []

for ii in range(n_community*n_members+1):
    tokens.append(
        chr(ord('A')+ii)
    )

#%%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, token_size=7):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_size, token_size)
        
    def forward(self, x, hw=None, short_term_memory=None):
        if hw is None:
            out, hw = self.rnn(x)
        else:   
            out, hw = self.rnn(x, hw)
            
        out = self.fc1(out[:,-1,:])
        return out, hw
    
    
#%%
class Dataset_converter(Dataset):
    def __init__(self, data, working_memory=1, short_term_memory=8):
        
        one_hot_encoded = np.zeros((len(data), len(tokens)), dtype=float)
        for ii, token in enumerate(data):
            one_hot_encoded[ii,ord(token)-65] = 1
        
        self.X = np.zeros((((len(data)-working_memory-short_term_memory)), short_term_memory, len(tokens)*working_memory))
        self.y = np.zeros((((len(data)-working_memory-short_term_memory)), len(tokens)))

        for ii in range(self.X.shape[0]):
            for jj in range(self.X.shape[1]):
                for kk in range(working_memory):
                    self.X[ii,jj,kk*len(tokens):(kk+1)*len(tokens)] = \
                    one_hot_encoded[ii+jj+kk,:]
                    
            self.y[ii] = \
                one_hot_encoded[ii+jj+kk+1,:]

        self.X = tnsr(self.X).float()
        self.y = tnsr(self.y).float()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
    
#%%

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bptt", type=int, help="bptt window")
    parser.add_argument("--node", type=int, help="node")

    args = parser.parse_args()

    reps = 10
    result_task1 = []
    result_task2 = []
    for rep in range(reps):
        ### initial training ###
        total_samples = 140000
        working_memory = 1
        short_term_memory = args.bptt
        node = args.node
        layer = 2
        token_size = n_community*n_members+1
        input_size = token_size*working_memory
        test_acc_task1 = []
        test_acc_task2 = []

        # output_sleep = len(tokens)
        input_size = len(tokens)*working_memory
        lr = 4e-4

        data_task1 = get_sequence(total_samples, n_community, n_members)
        data_set_task1 = Dataset_converter(data_task1, working_memory, short_term_memory)
        train_loader_task1 = DataLoader(data_set_task1, batch_size=1, shuffle=False)

        data_task2 = get_sequence(total_samples, n_community, n_members, train=False)
        data_set_task2 = Dataset_converter(data_task2, working_memory, short_term_memory)
        train_loader_task2 = DataLoader(data_set_task2, batch_size=1, shuffle=False)

        network1 = RNN(input_size, node, num_layers=layer, token_size=token_size)

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        correct_task1 = np.zeros(1000,dtype=float)
        correct_task2 = np.zeros(1000,dtype=float)
        for (X, y), (X_, y_) in zip(train_loader_task1, train_loader_task2):
            optimizer.zero_grad()
        
            if total == 0:
                predicted_y, hidden = network1(X)
            else:
                predicted_y, hidden = network1(X, hw=memory)

            
            loss = criterion(predicted_y, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                memory = hidden.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=1)

                if total == 0:
                    predicted_y, mem_ = network1(X_)
                else:
                    predicted_y, mem_ = network1(X_, hw=mem_)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y.argmax(axis=1)

                total += 1
                if true_y == estimated_y:
                    correct_task1[total%1000] = 1
                else:
                    correct_task1[total%1000] = 0

                if true_y_ == estimated_y_:
                        correct_task2[total%1000] = 1
                else:
                    correct_task2[total%1000] = 0

                test_acc_task1.append(
                    np.sum(correct_task1)/total if total<1000 else np.sum(correct_task1)/1000
                )
                test_acc_task2.append(
                    np.sum(correct_task2)/total if total<1000 else np.sum(correct_task2)/1000
                )

                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, task1 accuracy: {test_acc_task1[-1]:.4f}, task2 accuracy: {test_acc_task2[-1]:.4f}')


        #################################################################################
        data_task1 = get_sequence(total_samples, n_community, n_members)
        data_set_task1 = Dataset_converter(data_task1, working_memory, short_term_memory)
        train_loader_task1 = DataLoader(data_set_task1, batch_size=1, shuffle=False)

        data_task2 = get_sequence(total_samples, n_community, n_members, train=False)
        data_set_task2 = Dataset_converter(data_task2, working_memory, short_term_memory)
        train_loader_task2 = DataLoader(data_set_task2, batch_size=1, shuffle=False)

        network1 = RNN(input_size, node, num_layers=layer, token_size=token_size)

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        for (X, y), (X_, y_) in zip(train_loader_task2, train_loader_task1):
            optimizer.zero_grad()
        
            if total == 0:
                predicted_y, hidden = network1(X)
            else:
                predicted_y, hidden = network1(X, hw=memory)

            
            loss = criterion(predicted_y, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                memory = hidden.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=1)

                if total == 0:
                    predicted_y, mem_ = network1(X_)
                else:
                    predicted_y, mem_ = network1(X_, hw=mem_)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y.argmax(axis=1)

                total += 1
                if true_y == estimated_y:
                    correct_task2[total%1000] = 1
                else:
                    correct_task2[total%1000] = 0

                if true_y_ == estimated_y_:
                        correct_task1[total%1000] = 1
                else:
                    correct_task1[total%1000] = 0

                test_acc_task1.append(
                    np.sum(correct_task1)/1000
                )
                test_acc_task2.append(
                    np.sum(correct_task2)/1000
                )

                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, task1 accuracy: {test_acc_task1[-1]:.4f}, task2 accuracy: {test_acc_task2[-1]:.4f}')

        result_task1.append(test_acc_task1)
        result_task2.append(test_acc_task2)

    summary = (result_task1, result_task2)
    with open('pickle_files/chunking_naive_rnn_'+str(args.bptt)+'_'+str(args.node), 'wb') as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()