#%%
## Load my files ##
import sys
sys.path.append('..')
from utils import get_sequence

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
class brain(nn.Module):
    def __init__(self, input_size, hidden_wake_size, hidden_sleep_size, sleep_output_size, num_layers=2, num_layers_sleep=2):
        super(brain, self).__init__()

        self.rnn = nn.RNN(input_size+sleep_output_size, hidden_wake_size, num_layers, nonlinearity='relu', batch_first=True)
        self.sleep_rnn = nn.RNN(input_size, hidden_sleep_size, num_layers_sleep, nonlinearity='relu', batch_first=True)
        self.sleep_fc = nn.Linear(hidden_sleep_size, sleep_output_size)
        self.wake_fc = nn.Linear(hidden_wake_size, len(tokens))
        self.sleep_output_size = sleep_output_size

    def forward(self, x, x_=None, hw=None, hs=None, sleep=False):
        # print(x.shape, 'x')
        if sleep:
            if hs == None:
                out, hs = self.sleep_rnn(x_)
            else:
                out, hs = self.sleep_rnn(x_, hs)
            # print(out.shape)
            sleep_out = self.sleep_fc(out)
        else:
            sleep_out = torch.zeros((1,1,self.sleep_output_size))
            
        x = torch.cat((x,sleep_out), dim=2)
        
        if hw == None:
            out, hw = self.rnn(x)
        else:
            out, hw = self.rnn(x, hw)

        out = self.wake_fc(out)

        if sleep:
            return out, hw, hs
        else:
            return out, hw
    
#%%
class compressor(nn.Module):
    def __init__(self, input_size, hidden_compressor_size, num_layers=1):
        super(compressor, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_compressor_size, num_layers, nonlinearity='relu', batch_first=True)
        self.compressor_fc = nn.Linear(hidden_compressor_size, 2)

    def forward(self, x, hc=None):
        if hc == None:
            out, hc = self.rnn(x)
        else:
            out, hc = self.rnn(x, hc)

        out = self.compressor_fc(out)
        
        return out, hc
    
#%%
def compute_geodesic(hidden1, hidden2):

    total_layers = len(hidden1)
    w = 0

    for ii in range(total_layers):
        w_ = np.array(dist( hidden1[ii], hidden2[ii], 'cosine'))
        w += w_
           
    return w[0][0]/total_layers

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
class Dataset_converter_compressor(Dataset):
    def __init__(self, data, mask):
        total_sample = len(data)
        self.X = np.zeros((total_sample-2, len(tokens)))
        self.y = np.zeros((total_sample-2, 2))
        for ii in range(total_sample-2):
            token = data[ii]
            self.X[ii, ord(token)-65] = 1 
            self.y[ii,mask[ii]] = 1
            

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
        total_samples = 40000
        working_memory = 1
        short_term_memory = args.bptt
        hidden_wake_size = args.node
        hidden_compressor_size = 10
        hidden_sleep_size = args.node
        sleep_output_size = 20
        num_layers_wake = 1
        num_layers_sleep = 1
        # output_sleep = len(tokens)
        input_size = len(tokens)*working_memory
        lr = 4e-4
        test_acc_task1 = []
        test_acc_task2 = []

        data_task1 = get_sequence(total_samples, n_community, n_members)
        data_set_task1 = Dataset_converter(data_task1, working_memory, short_term_memory)
        train_loader_task1 = DataLoader(data_set_task1, batch_size=1, shuffle=False)

        data_task2 = get_sequence(total_samples, n_community, n_members, train=False)
        data_set_task2 = Dataset_converter(data_task2, working_memory, short_term_memory)
        train_loader_task2 = DataLoader(data_set_task2, batch_size=1, shuffle=False)

        network1 = brain(input_size, hidden_wake_size, hidden_sleep_size, sleep_output_size, num_layers_wake, num_layers_sleep)

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
                predicted_y, hidden = network1(X, hw=mem)
                
            loss = criterion(predicted_y[0], y)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                mem=hidden.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=2)


                if total == 0:
                    predicted_y, mem_ = network1(X_)
                else:
                    predicted_y, mem_ = network1(X_, hw=mem_)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y.argmax(axis=2)

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

        #%%
        compressor_sample = 20000

        data_compressor = get_sequence(compressor_sample, n_community, n_members)

        data_set_compressor = Dataset_converter(data_compressor, working_memory, short_term_memory)
        compressor_loader = DataLoader(data_set_compressor, batch_size=1, shuffle=False) 

        ii = 0
        dis = [0]
        # community = ''

        with torch.no_grad():
            for X, _ in compressor_loader:
                if ii==0:
                    id, hw = network1(X)
                    id_current = hw
                    # community = tokens[torch.argmax(X[0])]
                else:
                    id, hw = network1(X, hw=hw)
                    id_current = hw
                    if ii>=1:
                        dis.append(compute_geodesic(prev_id, id_current))
                        # print(dis)
                        # if dis[-1] >0.407:
                        #     # print(dis, tokens[torch.argmax(X[0])])
                        #     community += tokens[torch.argmax(X[0])]
                            
                    
                prev_id = id_current
                ii += 1
        #%%
        dis_array = np.array(dis)
        # threshold = np.quantile(dis_array, .8)
        # peaks = find_peaks(dis_array, .7)[0]
        peaks = [-100] 
        threshold = 0.3
        # prev_dis = 1

        for ii, dis in enumerate(dis_array):
            if dis >= threshold:
                if peaks[-1] == ii-1:
                    peaks.pop(-1)

                peaks.append(ii)
            
            # prev_dis = dis 

        peaks.pop(0)
        mask = np.zeros(dis_array.shape, dtype=int)
        mask[peaks] = 1
        # mask = ((dis_array>threshold)*1)
        print(mask[-100:])
        #%%
        data_set = Dataset_converter_compressor(data_compressor, mask)
        compressor_loader = DataLoader(data_set, batch_size=1, shuffle=False) 
        compression = []

        compressor_model = compressor(input_size, hidden_compressor_size)
        optimizer = torch.optim.SGD(compressor_model.parameters(), lr=4e-4, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        correct = np.zeros(1000, dtype=float)
        for X, y in compressor_loader:
            optimizer.zero_grad()

            if total == 0:
                predicted_y, hidden = compressor_model(X)
            else:
                predicted_y, hidden = compressor_model(X, hc=mem)
                
            loss = criterion(predicted_y, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mem = hidden.clone()

                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=1)

                if estimated_y[0]:
                    compression.append((true_y[0],estimated_y[0],tokens[X.argmax(axis=1)]))
                    
                total += 1
                if true_y == estimated_y:
                    correct[total%1000] = 1
                else:
                    correct[total%1000] = 0


        #%%
        sleep_samples = 100000
        data_sleep_task1 = get_sequence(sleep_samples, n_community, n_members)
        data_set_sleep_task1 = Dataset_converter(data_sleep_task1, working_memory, short_term_memory)
        sleep_loader_task1 = DataLoader(data_set_sleep_task1, batch_size=1, shuffle=False)

        data_sleep_task2 = get_sequence(sleep_samples, n_community, n_members, train=False)
        data_set_sleep_task2 = Dataset_converter(data_sleep_task2, working_memory, short_term_memory)
        sleep_loader_task2 = DataLoader(data_set_sleep_task2, batch_size=1, shuffle=False)
        # network1.rnn.requires_grad = True
        # network1.wake_fc.requires_grad = True

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        hidden_s = None
        correct_task1 = np.zeros(1000,dtype=float)
        correct_task2 = np.zeros(1000,dtype=float)
        for (X, y), (X_, y_) in zip(sleep_loader_task1, sleep_loader_task2):

            with torch.no_grad():
                if total == 0:
                    community = X.clone()
                    prev_community = X.clone()
                    predicted_y, hidden = compressor_model(X[0])
                else:
                    predicted_y, hidden = compressor_model(X[0], hc=hidden)

                selection = predicted_y.argmax(axis=1)


                if selection:        
                    community = prev_community.clone()
                    prev_community = X.clone()

            ##############################################
                if total == 0:
                    community_ = X_.clone()
                    prev_community_ = X_.clone()
                    predicted_y_, hidden_ = compressor_model(X_[0])
                else:
                    predicted_y_, hidden_ = compressor_model(X_[0], hc=hidden_)

                selection_ = predicted_y_.argmax(axis=1)


                if selection_:        
                    community_ = prev_community_.clone()
                    prev_community_ = X_.clone()
            ####################################################################
            optimizer.zero_grad()
            if total == 0:
                predicted_y, hidden_w, hidden_s = network1(X, community, sleep=True)
            else:
                predicted_y, hidden_w, hidden_s = network1(X, community, hw=mem, hs=mem_, sleep=True)
                
            loss = criterion(predicted_y[0], y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mem=hidden_w.clone()
                mem_=hidden_s.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=2)

                if total == 0:
                    predicted_y_, hidden_w_, hidden_s_ = network1(X_, community_, sleep=True)
                else:
                    predicted_y_, hidden_w_, hidden_s_ = network1(X_, community_, hw=hidden_w_, hs=hidden_s_, sleep=True)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y_.argmax(axis=2)

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


    #################################################################################################################################################################

        data_task1 = get_sequence(total_samples, n_community, n_members)
        data_set_task1 = Dataset_converter(data_task1, working_memory, short_term_memory)
        train_loader_task1 = DataLoader(data_set_task1, batch_size=1, shuffle=False)

        data_task2 = get_sequence(total_samples, n_community, n_members, train=False)
        data_set_task2 = Dataset_converter(data_task2, working_memory, short_term_memory)
        train_loader_task2 = DataLoader(data_set_task2, batch_size=1, shuffle=False)

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        correct_task1 = np.zeros(1000,dtype=float)
        correct_task2 = np.zeros(1000,dtype=float)
        for (X, y), (X_, y_) in zip(train_loader_task2, train_loader_task1):
            optimizer.zero_grad()

            if total == 0:
                predicted_y, hidden = network1(X)
            else:
                predicted_y, hidden = network1(X, hw=mem)
                
            loss = criterion(predicted_y[0], y)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                mem=hidden.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=2)


                if total == 0:
                    predicted_y, mem_ = network1(X_)
                else:
                    predicted_y, mem_ = network1(X_, hw=mem_)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y.argmax(axis=2)

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
                    np.sum(correct_task1)/total if total<1000 else np.sum(correct_task1)/1000
                )
                test_acc_task2.append(
                    np.sum(correct_task2)/total if total<1000 else np.sum(correct_task2)/1000
                )

                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, task1 accuracy: {test_acc_task1[-1]:.4f}, task2 accuracy: {test_acc_task2[-1]:.4f}')

        #%%
        # compressor_sample = 20000

        # data_compressor = get_sequence(compressor_sample, n_community, n_members, train=False)

        # data_set_compressor = Dataset_converter(data_compressor, working_memory, short_term_memory)
        # compressor_loader = DataLoader(data_set_compressor, batch_size=1, shuffle=False) 

        # ii = 0
        # dis = [0]
        # # community = ''

        # with torch.no_grad():
        #     for X, _ in compressor_loader:
        #         if ii==0:
        #             id, hw = network1(X)
        #             id_current = hw
        #             # community = tokens[torch.argmax(X[0])]
        #         else:
        #             id, hw = network1(X, hw=hw)
        #             id_current = hw
        #             if ii>=1:
        #                 dis.append(compute_geodesic(prev_id, id_current))
        #                 # print(dis)
        #                 # if dis[-1] >0.407:
        #                 #     # print(dis, tokens[torch.argmax(X[0])])
        #                 #     community += tokens[torch.argmax(X[0])]
                            
                    
        #         prev_id = id_current
        #         ii += 1
        # #%%
        # dis_array = np.array(dis)
        # # threshold = np.quantile(dis_array, .8)
        # # peaks = find_peaks(dis_array, .7)[0]
        # peaks = [-100] 
        # threshold = 0.3
        # # prev_dis = 1

        # for ii, dis in enumerate(dis_array):
        #     if dis >= threshold:
        #         if peaks[-1] == ii-1:
        #             peaks.pop(-1)

        #         peaks.append(ii)
            
        #     # prev_dis = dis 

        # peaks.pop(0)
        # mask = np.zeros(dis_array.shape, dtype=int)
        # mask[peaks] = 1
        # # mask = ((dis_array>threshold)*1)
        # print(mask[-100:])
        # #%%
        # data_set = Dataset_converter_compressor(data_compressor, mask)
        # compressor_loader = DataLoader(data_set, batch_size=1, shuffle=False) 
        # compression = []

        # compressor_model = compressor(input_size, hidden_compressor_size)
        # optimizer = torch.optim.SGD(compressor_model.parameters(), lr=4e-4, momentum=0.95)
        # criterion = torch.nn.CrossEntropyLoss()

        # total = 0
        # correct = np.zeros(1000, dtype=float)
        # for X, y in compressor_loader:
        #     optimizer.zero_grad()

        #     if total == 0:
        #         predicted_y, hidden = compressor_model(X)
        #     else:
        #         predicted_y, hidden = compressor_model(X, hc=mem)
                
        #     loss = criterion(predicted_y, y)
        #     loss.backward()
        #     optimizer.step()

        #     with torch.no_grad():
        #         mem = hidden.clone()

        #         true_y = y.argmax(axis=1)
        #         estimated_y = predicted_y.argmax(axis=1)

        #         if estimated_y[0]:
        #             compression.append((true_y[0],estimated_y[0],tokens[X.argmax(axis=1)]))
                    
        #         total += 1
        #         if true_y == estimated_y:
        #             correct[total%1000] = 1
        #         else:
        #             correct[total%1000] = 0


        #%%
        sleep_samples = 100000
        data_sleep_task1 = get_sequence(sleep_samples, n_community, n_members)
        data_set_sleep_task1 = Dataset_converter(data_sleep_task1, working_memory, short_term_memory)
        sleep_loader_task1 = DataLoader(data_set_sleep_task1, batch_size=1, shuffle=False)

        data_sleep_task2 = get_sequence(sleep_samples, n_community, n_members, train=False)
        data_set_sleep_task2 = Dataset_converter(data_sleep_task2, working_memory, short_term_memory)
        sleep_loader_task2 = DataLoader(data_set_sleep_task2, batch_size=1, shuffle=False)
        # network1.rnn.requires_grad = True
        # network1.wake_fc.requires_grad = True

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        hidden_s = None
        correct_task1 = np.zeros(1000,dtype=float)
        correct_task2 = np.zeros(1000,dtype=float)
        for (X, y), (X_, y_) in zip(sleep_loader_task2, sleep_loader_task1):

            with torch.no_grad():
                if total == 0:
                    community = X.clone()
                    prev_community = X.clone()
                    predicted_y, hidden = compressor_model(X[0])
                else:
                    predicted_y, hidden = compressor_model(X[0], hc=hidden)

                selection = predicted_y.argmax(axis=1)


                if selection:        
                    community = prev_community.clone()
                    prev_community = X.clone()

            ##############################################
                if total == 0:
                    community_ = X_.clone()
                    prev_community_ = X_.clone()
                    predicted_y_, hidden_ = compressor_model(X_[0])
                else:
                    predicted_y_, hidden_ = compressor_model(X_[0], hc=hidden_)

                selection_ = predicted_y_.argmax(axis=1)


                if selection_:        
                    community_ = prev_community_.clone()
                    prev_community_ = X_.clone()
            ####################################################################
            optimizer.zero_grad()
            if total == 0:
                predicted_y, hidden_w, hidden_s = network1(X, community, sleep=True)
            else:
                predicted_y, hidden_w, hidden_s = network1(X, community, hw=mem, hs=mem_, sleep=True)
                
            loss = criterion(predicted_y[0], y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mem=hidden_w.clone()
                mem_=hidden_s.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=2)

                if total == 0:
                    predicted_y_, hidden_w_, hidden_s_ = network1(X_, community_, sleep=True)
                else:
                    predicted_y_, hidden_w_, hidden_s_ = network1(X_, community_, hw=hidden_w_, hs=hidden_s_, sleep=True)

                true_y_ = y_.argmax(axis=1)
                estimated_y_ = predicted_y_.argmax(axis=2)

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
                    np.sum(correct_task1)/total if total<1000 else np.sum(correct_task1)/1000
                )
                test_acc_task2.append(
                    np.sum(correct_task2)/total if total<1000 else np.sum(correct_task2)/1000
                )

                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, task1 accuracy: {test_acc_task1[-1]:.4f}, task2 accuracy: {test_acc_task2[-1]:.4f}')

        result_task1.append(test_acc_task1)
        result_task2.append(test_acc_task2)
    
    summary = (result_task1, result_task2)
    with open('pickle_files/chunking_CL_'+str(args.bptt)+'_'+str(args.node), 'wb') as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()