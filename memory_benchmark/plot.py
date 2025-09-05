#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
#%%
with open('../pickle_files/memory_capacity_random.pickle', 'rb') as f:
    df_random = pickle.load(f)
# %%
with open('../pickle_files/memory_capacity_patterned.pickle', 'rb') as f:
    df_linear = pickle.load(f)
# %%
with open('../pickle_files/memory_capacity_hard_patterned.pickle', 'rb') as f:
    df = pickle.load(f)

#%%
with open('../pickle_files/memory_capacity_random_autoencoder.pickle', 'rb') as f:
    df_random_en = pickle.load(f)
# %%
with open('../pickle_files/memory_capacity_patterned_autoencoder.pickle', 'rb') as f:
    df_linear_en = pickle.load(f)
# %%
with open('../pickle_files/memory_capacity_hard_patterned_autoencoder.pickle', 'rb') as f:
    df_en = pickle.load(f)
# %%
result_random = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result_linear = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result_random_en = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result_linear_en = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result_en = {'2':[], '4':[], '6':[], '8':[], '10':[]}
# %%
reps = 5
bptts = [2,4,6,8]
for rep in range(reps):
    for bptt in bptts:
        result_random[str(bptt)].append(
            list(
                1 - df_random[(df_random['BBPTT'] == bptt) & (df_random['reps'] == rep)]['accuracy']
            )
        )

        result_linear[str(bptt)].append(
            list(
                1 - df_linear[(df_linear['BBPTT'] == bptt) & (df_linear['reps'] == rep)]['accuracy']
            )
        )

        result[str(bptt)].append(
            list(
                1 - df[(df['BBPTT'] == bptt) & (df['reps'] == rep)]['accuracy']
            )
        )

        result_random_en[str(bptt)].append(
            list(
                1 - df_random_en[(df_random_en['BBPTT'] == bptt) & (df_random_en['reps'] == rep)]['accuracy']
            )
        )

        result_linear_en[str(bptt)].append(
            list(
                1 - df_linear_en[(df_linear_en['BBPTT'] == bptt) & (df_linear_en['reps'] == rep)]['accuracy']
            )
        )

        result_en[str(bptt)].append(
            list(
                1 - df_en[(df_en['BBPTT'] == bptt) & (df_en['reps'] == rep)]['accuracy']
            )
        )
# %%
fig, ax = plt.subplots(2, 3, figsize=(24, 12), sharey=True, sharex=True)
sns.set_context('talk')
fontsize=40

past_recalls = [1,2,3,4,5]
for bptt in bptts:
    ax[0][0].plot(past_recalls, np.mean(result_random[str(bptt)], axis=0), '-o', label='context length '+ str(bptt), linewidth=4)
    ax[0][0].fill_between(past_recalls, np.quantile(result_random[str(bptt)], [0.25], axis=0)[0], np.quantile(result_random[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[0][1].plot(past_recalls, np.mean(result[str(bptt)], axis=0), '-o', linewidth=4)
    ax[0][1].fill_between(past_recalls, np.quantile(result[str(bptt)], [0.25], axis=0)[0], np.quantile(result[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[0][2].plot(past_recalls, np.mean(result_linear[str(bptt)], axis=0), '-o', linewidth=4)
    ax[0][2].fill_between(past_recalls, np.quantile(result_linear[str(bptt)], [0.25], axis=0)[0], np.quantile(result_linear[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    
    ax[1][0].plot(past_recalls, np.mean(result_random_en[str(bptt)], axis=0), '-o', linewidth=4)
    ax[1][0].fill_between(past_recalls, np.quantile(result_random_en[str(bptt)], [0.25], axis=0)[0], np.quantile(result_random_en[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[1][1].plot(past_recalls, np.mean(result_en[str(bptt)], axis=0), '-o', linewidth=4)
    ax[1][1].fill_between(past_recalls, np.quantile(result_en[str(bptt)], [0.25], axis=0)[0], np.quantile(result_en[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[1][2].plot(past_recalls, np.mean(result_linear_en[str(bptt)], axis=0), '-o', linewidth=4)
    ax[1][2].fill_between(past_recalls, np.quantile(result_linear_en[str(bptt)], [0.25], axis=0)[0], np.quantile(result_linear_en[str(bptt)], [0.75], axis=0)[0], alpha=.3)

ax[0][0].hlines(1-1/7.0, 1, 5, linestyle='--', color='black', linewidth=2, label='chance')

for jj in range(2):
    for ii in range(3):
        ax[jj][ii].hlines(1-1/7.0, 1, 5, linestyle='--', color='black', linewidth=2)

        ax[jj][ii].set_yticks([0,.5,1])
        right_side = ax[jj][ii].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii].spines["top"]
        top_side.set_visible(False)

        ax[jj][ii].tick_params(labelsize=fontsize-10)
        ax[jj][ii].tick_params(labelsize=fontsize-10)
        ax[jj][ii].tick_params(labelsize=fontsize-10)

ax[0][0].set_title('Random', fontsize=fontsize)
ax[0][1].set_title('Non-linear', fontsize=fontsize)
ax[0][2].set_title('Linear', fontsize=fontsize)

# ax[1][1].set_title('Learn Pattern + Recall', fontsize=fontsize)

ax[1][1].set_xlabel('How far in the past is recalled', fontsize=fontsize-5)
ax[1][0].set_ylabel('')

fig.text(0.06,.3,'Reconstruction Error', fontsize=fontsize-5, rotation=90)
fig.text(0.0,.55,'(a) Learn Pattern', fontsize=fontsize, rotation=90)
fig.text(0.0,.1,'(b) Learn Pattern \n    + Recall', fontsize=fontsize, rotation=90)

leg = fig.legend(bbox_to_anchor=(.9, 0), bbox_transform=plt.gcf().transFigure,
                        ncol=5, fontsize=25, frameon=False)

plt.savefig('../plots/memory_capacity_RNN.pdf', bbox_inches='tight')
# %%
