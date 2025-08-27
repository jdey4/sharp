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
# %%
result_random = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result_linear = {'2':[], '4':[], '6':[], '8':[], '10':[]}
result = {'2':[], '4':[], '6':[], '8':[], '10':[]}
# %%
reps = 10
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
# %%
fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharey=True, sharex=True)
sns.set_context('talk')
fontsize=40

past_recalls = [1,2,3,4,5]
for bptt in bptts:
    ax[0].plot(past_recalls, np.mean(result_random[str(bptt)], axis=0), '-o', label='BPTT '+ str(bptt), linewidth=4)
    ax[0].fill_between(past_recalls, np.quantile(result_random[str(bptt)], [0.25], axis=0)[0], np.quantile(result_random[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[1].plot(past_recalls, np.mean(result[str(bptt)], axis=0), '-o', linewidth=4)
    ax[1].fill_between(past_recalls, np.quantile(result[str(bptt)], [0.25], axis=0)[0], np.quantile(result[str(bptt)], [0.75], axis=0)[0], alpha=.3)
    ax[2].plot(past_recalls, np.mean(result_linear[str(bptt)], axis=0), '-o', linewidth=4)
    ax[2].fill_between(past_recalls, np.quantile(result_linear[str(bptt)], [0.25], axis=0)[0], np.quantile(result_linear[str(bptt)], [0.75], axis=0)[0], alpha=.3)

for ii in range(3):
    ax[ii].set_yticks([0,.5,.8])
    right_side = ax[ii].spines["right"]
    right_side.set_visible(False)
    top_side = ax[ii].spines["top"]
    top_side.set_visible(False)

ax[0].tick_params(labelsize=fontsize-10)
ax[1].tick_params(labelsize=fontsize-10)
ax[2].tick_params(labelsize=fontsize-10)

ax[0].set_title('Random', fontsize=fontsize)
ax[1].set_title('Non-linear', fontsize=fontsize)
ax[2].set_title('Linear', fontsize=fontsize)

ax[1].set_xlabel('How far in the past is recalled', fontsize=fontsize)
ax[0].set_ylabel('Reconstruction Error', fontsize=fontsize)

leg = fig.legend(bbox_to_anchor=(.8, -0.1), bbox_transform=plt.gcf().transFigure,
                        ncol=4, fontsize=30, frameon=False)

plt.savefig('../plots/memory_capacity_RNN.pdf', bbox_inches='tight')
# %%
