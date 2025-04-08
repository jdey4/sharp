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
        # self.connect_sleep = nn.Linear(hidden_wake_size, 3)
        self.sleep_rnn = nn.RNN(hidden_wake_size, hidden_sleep_size, num_layers_sleep, nonlinearity='relu', batch_first=True)
        self.sleep_fc = nn.Linear(hidden_sleep_size, sleep_output_size)
        self.wake_fc = nn.Linear(hidden_wake_size, len(tokens))
        self.sleep_output_size = sleep_output_size

    def forward(self, x, x_=None, hw=None, hs=None, sleep=False):
        # print(x.shape, 'x')
        if sleep:
            # x_ = self.connect_sleep(x_)
            
            if hs == None:
                out, hs = self.sleep_rnn(x_)
            else:
                out, hs = self.sleep_rnn(x_, hs)
            # print(out.shape)
            sleep_out = self.sleep_fc(out)
            # print(sleep_out.size(), x.size())
        else:
            sleep_out = torch.zeros((1,x.size(1),self.sleep_output_size))
            
        # print(x.size())
        x = torch.cat((x,sleep_out), dim=2)
        
        if hw == None:
            out, hw = self.rnn(x)
        else:
            out, hw = self.rnn(x, hw)

        out = self.wake_fc(out[:,-1,:])

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
        
        self.X = np.zeros((((len(data)-short_term_memory)), short_term_memory, len(tokens)*working_memory))
        self.y = np.zeros((((len(data)-short_term_memory)), len(tokens)))

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
    result = []

    for _ in range(reps):
        ### initial training ###
        total_samples = 40000
        working_memory = 1
        short_term_memory = args.bptt
        hidden_wake_size = args.node
        hidden_sleep_size = args.node
        sleep_output_size = 20
        num_layers_wake = 1
        num_layers_sleep = 1
        # output_sleep = len(tokens)
        input_size = len(tokens)*working_memory
        lr = 3e-4
        test_acc = []

        data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)

        data_set = Dataset_converter(data, working_memory, short_term_memory)
        train_loader = DataLoader(data_set, batch_size=1, shuffle=False)

        network1 = brain(input_size, hidden_wake_size, hidden_sleep_size, sleep_output_size, num_layers_wake, num_layers_sleep)

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        correct = np.zeros(1000,dtype=float)
        for X, y in train_loader:
            optimizer.zero_grad()

            if total == 0:
                predicted_y, hidden = network1(X)
            else:
                predicted_y, hidden = network1(X, hw=mem)
            
            # print(predicted_y.shape, y.shape)
            loss = criterion(predicted_y, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                mem=hidden.clone()
                true_y = y.argmax(axis=1)
                estimated_y = predicted_y.argmax(axis=1)

                total += 1
                if true_y == estimated_y:
                        correct[total%1000] = 1
                else:
                    correct[total%1000] = 0

                test_acc.append(
                    np.sum(correct)/total if total<1000 else np.sum(correct)/1000
                )
                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, accuracy: {test_acc[-1]:.4f}')
        #%%
        centroids = []

        threshold = .65
        n_samples = 10000
        idx = torch.randint(0, len(tokens), (1,)) [0]
        X_hat = torch.zeros(len(tokens),dtype=torch.float32)
        X_hat[idx] = 1.0

        for ii in range(n_samples):
            if ii == 0:
                # seq += tokens[idx]        
                X_hat, mem = network1(X_hat.reshape(1,1,-1))
                centroids.append(mem)
            else:
                X_hat, mem = network1(X_hat, mem)

            dis = []
            min_dis = 10
            min_dis_id = -1
            for jj in range(len(centroids)):
                dis.append(
                    compute_geodesic(centroids[jj].detach().numpy(), mem.detach().numpy())
                )
                if min_dis >= dis[-1]:
                    min_dis = dis[-1] 
                    min_dis_id = jj 
            if min_dis < threshold:
                centroids[min_dis_id] = (centroids[min_dis_id] + mem)/2.0
            else:
                centroids.append(mem)

            # print(min_dis)   
            X_hat = torch.nn.functional.softmax(X_hat, dim=1)
            dist_categ = torch.distributions.Categorical(probs=X_hat.reshape(-1))
            idx = dist_categ.sample()

            X_hat = torch.zeros(len(tokens),dtype=torch.float32)
            X_hat[idx] = 1.0
            X_hat = X_hat.reshape(1,1,-1)   
            
        #%%
        sleep_samples = 200000
        data_sleep = get_sequence(sleep_samples, n_community, n_members, train_percent=1.0)
        data_set_sleep = Dataset_converter(data_sleep, working_memory, short_term_memory)

        sleep_loader = DataLoader(data_set_sleep, batch_size=1, shuffle=False)

        network1.rnn.requires_grad = True
        network1.wake_fc.requires_grad = True

        optimizer = torch.optim.SGD(network1.parameters(), lr=lr, momentum=0.95)
        criterion = torch.nn.CrossEntropyLoss()

        total = 0
        hidden_s = None
        # correct = np.zeros(1000,dtype=float)
        communities = [0]
        current_community = 0
        community = torch.zeros((1, short_term_memory, centroids[current_community].size(2)))
        prev_community = torch.zeros((1, short_term_memory, centroids[current_community].size(2)))
        
        for X, y in sleep_loader:
            optimizer.zero_grad()
            if total == 0:
                predicted_y, hidden_w = network1(X)
            elif total==1:
                predicted_y, hidden_w, hidden_s = network1(X, prev_community, hw=mem, sleep=True)
            else:
                predicted_y, hidden_w, hidden_s = network1(X, prev_community, hw=mem, hs=mem_, sleep=True)

            loss = criterion(predicted_y, y)
            loss.backward(retain_graph=True)
            optimizer.step()



            with torch.no_grad():
                dis = []
                for center in centroids:
                    dis.append(
                        compute_geodesic(center.detach().numpy(), hidden_w.detach().numpy())
                    )
                
                idx = np.argmin(dis)

                if idx != communities[-1]:
                    communities.append(idx)  
                    current_community = communities[-1]
                    prev_community = community.clone()
                else:
                    community[0][0:short_term_memory-1] = community[0][1:short_term_memory].clone()
                    community[0][-1] = centroids[current_community].view(1,1,-1)[0].clone()
                
                mem=hidden_w.detach().clone()

                if total != 0:
                    mem_=hidden_s.clone()

                true_y = y.argmax(axis=1)
                # print(predicted_y)
                estimated_y = predicted_y.argmax(axis=1)

                total += 1
                if true_y == estimated_y:
                        correct[total%1000] = 1
                else:
                    correct[total%1000] = 0

                test_acc.append(
                    np.sum(correct)/1000
                )
                if total%1000 == 0:
                    print(f'Iter : {total+1}, loss: {loss:.4f}, accuracy: {test_acc[-1]:.4f}')

        result.append(test_acc)

    with open('pickle_files/clustering_'+str(args.bptt)+'_'+str(args.node), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()


