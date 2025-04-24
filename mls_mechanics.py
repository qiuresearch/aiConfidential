#%%
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
from scipy.stats import gaussian_kde
import seaborn as sns
import plotly.graph_objects as go
import glob
import tqdm

import matplotlib.pyplot as plt
import numpy as np

import glance_midat
import importlib
importlib.reload(glance_midat)

def resize_matrix_grps(matrix, grp_sizes, method='up', dims=[0]):
    """ size up each group in the matrix to match the largest group,  
    assuming rows are ordered as in grp_sizes"""

    grp_iranges = np.cumsum(grp_sizes)
    grp_iranges = np.insert(grp_iranges, 0, 0)

    if isinstance(method, int):
        fixed_size = method
    elif method.lower() == 'down':
        fixed_size = min(grp_sizes)
    elif method.lower() == 'up':
        fixed_size = max(grp_sizes)

    resized_idx = []
    for i in range(len(grp_sizes)):
        grp_idx = np.arange(grp_iranges[i], grp_iranges[i+1])
        if grp_sizes[i] == 0: continue
        if fixed_size > grp_sizes[i]:
            num_tiles = np.ceil(fixed_size / grp_sizes[i])
            grp_idx = np.repeat(grp_idx, num_tiles) # ([1,2,3], 2) -> [1,1,2,2,3,3]
            resized_idx.extend(grp_idx[:fixed_size])
        elif fixed_size < grp_sizes[i]:
            # if the group size is larger than fixed_size, randomly select fixed_size samples
            grp_idx = np.random.choice(grp_idx, fixed_size, replace=False)
            resized_idx.extend(grp_idx)
        else:
            resized_idx.extend(grp_idx)

    # apply the resized index to the matrix on all axes
    for idim in dims:
        if idim == 0:
            matrix = matrix[resized_idx]
        else:
            matrix = np.swapaxes(matrix, idim, 0)
            matrix = matrix[resized_idx]
            matrix = np.swapaxes(matrix, 0, idim)

    return matrix, [fixed_size] * len(grp_sizes)


def plot_pairwise_matrix(matrix, title=None, xlabel='x', ylabel='y', figsize=None,
                         cmap='magma', # 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                         row_grp_sizes=None, col_grp_sizes=None,
                         row_grp_names=None, col_grp_names=None,
                         vmin=0, vmax=1, resize=False,
                         diagonal=True, hline=True, vline=True, average=True,
                         ):

    if figsize is not None:
        plt.figure(figsize=figsize)

    if resize:
        if row_grp_sizes is not None:
            matrix, row_grp_sizes = resize_matrix_grps(matrix, row_grp_sizes, method=resize)
        if col_grp_sizes is not None:
            matrix, col_grp_sizes = resize_matrix_grps(matrix.T, col_grp_sizes, method=resize)
            matrix = matrix.T

    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # plot dashed lines for the group boundaries
    if row_grp_sizes is not None:
        row_grp_iranges = np.cumsum(row_grp_sizes)
        if hline:
            for i in range(len(row_grp_iranges) - 1):
                plt.axhline(y=row_grp_iranges[i], color='r', linestyle='--', linewidth=0.5)
        if row_grp_names is not None:
            row_grp_iranges = np.insert(row_grp_iranges, 0, 0)
            for i, name in enumerate(row_grp_names):
                plt.text(-0.5, row_grp_iranges[i] + row_grp_sizes[i] / 2, name, color='r', fontsize=8,
                        ha='right', va='center', rotation=-45)
            
    if col_grp_sizes is not None:
        col_grp_iranges = np.cumsum(col_grp_sizes)
        if vline:
            for i in range(len(col_grp_iranges) - 1):
                plt.axvline(x=col_grp_iranges[i], color='r', linestyle='--', linewidth=0.5)
        if col_grp_names is not None:
            col_grp_iranges = np.insert(col_grp_iranges, 0, 0)
            for i, name in enumerate(col_grp_names):
                plt.text(col_grp_iranges[i] + col_grp_sizes[i] / 2, -0.5, name, color='r', fontsize=8,
                        ha='center', va='bottom', rotation=45)
                plt.text(col_grp_iranges[i] + col_grp_sizes[i] / 2, matrix.shape[0]+0.5, name, color='r', fontsize=8,
                        ha='center', va='top', rotation=45)
                
    # display the mean of each row and col group
    if average and row_grp_sizes is not None and col_grp_sizes is not None:
        for i in range(len(row_grp_sizes)):
            if row_grp_sizes[i] <= 0: continue
            for j in range(len(col_grp_sizes)):
                if col_grp_sizes[j] <= 0: continue
                block_data = matrix[row_grp_iranges[i]:row_grp_iranges[i+1], col_grp_iranges[j]:col_grp_iranges[j+1]]
                # get the mean of the block without NaN
                block_data = block_data[~np.isnan(block_data)]
                if block_data.size == 0: continue
                plt.text(col_grp_iranges[j] + col_grp_sizes[j] / 2,
                         row_grp_iranges[i] + row_grp_sizes[i] / 2,
                         f'{block_data.mean():.3f}', color='g', fontsize=8,
                         ha='center', va='center', rotation=0)

    # draw a diagonal line from (0, 0) to (len(tr_df0), len(ts_df0))
    if diagonal:
        plt.plot([0, matrix.shape[1]-1], [0, matrix.shape[0]-1], color='r', linestyle='--', linewidth=0.5)

    # disable aspect ratio
    plt.gca().set_aspect('auto')
    plt.colorbar()
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    # only show the ticks for the first and last row/col
    plt.xticks([0, matrix.shape[1]-1], [0, matrix.shape[1]-1])
    plt.yticks([0, matrix.shape[0]-1], [0, matrix.shape[0]-1])
    plt.show()


def collate_sort_df(df, by='moltype', ascending=False):
    """ collate each group of the metric and sort by the number of samples in each group"""

    df_grps = []
    grp_sizes = []
    grp_names = []
    for grp, df_grp in df.groupby(by):
        grp_names.append(grp)
        grp_sizes.append(df_grp.shape[0])
        df_grps.append(df_grp)

    if ascending:
        sorted_idx = np.argsort(grp_sizes)
    else:
        sorted_idx = np.argsort(grp_sizes)[::-1]

    df = pd.concat([df_grps[i] for i in sorted_idx], ignore_index=True)
    grp_sizes = [grp_sizes[i] for i in sorted_idx]
    grp_names = [grp_names[i] for i in sorted_idx]

    return df, grp_sizes, grp_names


def get_dfcol2irow(df, key='idx'):
    """ get the irow to the key in the df"""
    key_values = df[key].values
    key2irow = np.full(key_values.max() + 1, -1, dtype=int)
    key2irow[key_values] = np.arange(df.shape[0])
    return key2irow


def collect_rand50_metrics(
        folder_path=os.path.join(os.path.expanduser('~'), 'bench/aiConfidential/data'),
        metric='f1',
        upsample=False,
        model_cutoff=0.8):
    # vl_files = glob.glob('**/*upsample*/*_VL.eval/eval_loss_meta.csv', recursive=True)
    vl_files = glob.glob(os.path.join(folder_path, '**/*/*_VL.eval/eval_loss_meta.csv'), recursive=True)
    if upsample:
        vl_files = [_s for _s in vl_files if 'upsample' in _s]
    else:
        vl_files = [_s for _s in vl_files if 'upsample' not in _s]
    print(f'Number of VL files (upsample={upsample}): {len(vl_files)}')

    # load the first VL file and its corresponding TR file
    vl_df = pd.read_csv(vl_files[0])
    tr_df = pd.read_csv(vl_files[0].replace('_VL.eval', '_TR.eval'))
    meta_df = pd.concat([vl_df, tr_df], ignore_index=True)
    num_samples = vl_df.shape[0] + tr_df.shape[0]
    print(f'vl_df shape: {vl_df.shape[0]}; tr_df shape: {tr_df.shape[0]}; num_samples: {num_samples}')

    # get the idx_map
    meta_df, grp_sizes, grp_names = collate_sort_df(meta_df)
    meta_df = meta_df.set_index('idx', drop=False)
    meta_df.index.name = 'index'
    idx2i = get_dfcol2irow(meta_df, key='idx')

    # group by "moltype" and get the idx list for each group
    # idx_per_moltype ={}
    # for grp, df_grp in meta_df.groupby('moltype'):
    #     idx_per_moltype[grp] = df_grp['idx'].values

    tr_vl_metrics = np.zeros((num_samples, len(vl_files)))
    vl_isa = np.zeros((num_samples, len(vl_files)))
    for i, vl_csv in tqdm.tqdm(enumerate(vl_files)):
        vl_df = pd.read_csv(vl_csv)

        i2mat = idx2i[vl_df['idx'].values]
        tr_vl_metrics[i2mat, i] = vl_df[metric].values
        vl_isa[i2mat, i] = 1 # 1 if in validation set, 0 if in training set
        
        tr_csv = vl_csv.replace('_VL.eval', '_TR.eval')
        tr_df = pd.read_csv(tr_csv)
        i2mat = idx2i[tr_df['idx'].values]
        tr_vl_metrics[i2mat, i] = tr_df[metric].values

    if model_cutoff is not None:
        # remove the models with mean f1 score < model_cutoff
        tr_vl_mean_per_model = np.mean(tr_vl_metrics, axis=0)
        # find index of models with mean f1 score < model_cutoff
        model_idx = np.where(tr_vl_mean_per_model <= model_cutoff)[0]
        if len(model_idx) > 0:
            print(f'Removing {len(model_idx)} models with mean f1 score < {model_cutoff}')
            tr_vl_metrics = np.delete(tr_vl_metrics, model_idx, axis=1)
            vl_isa = np.delete(vl_isa, model_idx, axis=1)
            vl_files = [vl_files[i] for i in range(len(vl_files)) if i not in model_idx]

    meta_df['vl_avg'] = np.sum(tr_vl_metrics * vl_isa, axis=1) / np.sum(vl_isa, axis=1)
    meta_df['tr_avg'] = np.sum(tr_vl_metrics * (1 - vl_isa), axis=1) / np.sum(1 - vl_isa, axis=1)

    meta_df['mem_score'] = (meta_df['tr_avg'] - meta_df['vl_avg'])
    meta_df['mem_score_normalized'] = meta_df['mem_score'] / (meta_df['tr_avg'] + meta_df['vl_avg'])    
    
    tr_isa = 1 - vl_isa
    influ_ij = np.zeros((len(meta_df), len(meta_df)))
    for j in range(len(meta_df)):
        influ_ij[:, j] = np.sum(tr_isa * tr_vl_metrics[j:j+1, :] * vl_isa[j:j+1, :], axis=1) / (np.sum(tr_isa * vl_isa[j:j+1, :], axis=1) + 1e-7) -\
                         np.sum(vl_isa * tr_vl_metrics[j:j+1, :] * vl_isa[j:j+1, :], axis=1) / (np.sum(vl_isa * vl_isa[j:j+1, :], axis=1) + 1e-7)
                             
    # fill the diagonal with meta_df['mem_score']
    influ_ij[np.diag_indices_from(influ_ij)] = meta_df['mem_score'].values

    # the code below takes too much memroy
    # tr_isa = 1 - vl_isa
    # vl_ijk = vl_isa[:, None, :] * f1_mat[None, :, :]
    # tr_ijk = tr_isa[:, None, :] * f1_mat[None, :, :]
    # influence_ij = np.sum(vl_ijk, axis=2) / np.sum(vl_isa[:, None, :], axis=2) -\
    #                np.sum(tr_ijk, axis=2) / np.sum(tr_isa[:, None, :], axis=2)

    return meta_df, tr_vl_metrics, vl_isa, influ_ij, grp_sizes, grp_names

#%% Load rand50 metrics
os.chdir('/home/xqiu/bench/aiConfidential')

meta_df, tr_vl_metrics, vl_isa, influ_ij, grp_sizes, grp_names = collect_rand50_metrics(model_cutoff=0.8)
meta_df.to_pickle('rand50_meta.pkl')

#%% Show TR and VL means for each model
tr_metric_mean_per_model = np.sum(tr_vl_metrics * (1 - vl_isa), axis=0) / np.sum(1 - vl_isa, axis=0)
vl_metric_mean_per_model = np.sum(tr_vl_metrics * vl_isa, axis=0) / np.sum(vl_isa, axis=0)
print(f'Mean TR metric per model: {tr_metric_mean_per_model.mean():.3f} +/- {tr_metric_mean_per_model.std():.3f}')
print(f'Mean VL metric per model: {vl_metric_mean_per_model.mean():.3f} +/- {vl_metric_mean_per_model.std():.3f}')
#%% Plot tr_metric_mean_per_model vs vl_metric_mean_per_model with matplotlib
plt.scatter(tr_metric_mean_per_model, vl_metric_mean_per_model, s=30)
plt.xlabel('TR metric mean per model')
plt.ylabel('VL metric mean per model')
plt.title('TR vs VL metric mean per model')
plt.show()

#%% Plot tr_vl_metrics
plot_pairwise_matrix(tr_vl_metrics, title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes, col_grp_sizes=None,
                    row_grp_names=grp_names, col_grp_names=None,
                    vmin=0, vmax=1.0, resize='up', diagonal=False,)

#%% Plot vl_isa
plot_pairwise_matrix(vl_isa, title='VL isa', xlabel='Rand50 Model', ylabel='Train-Valid', cmap='gray',
                    row_grp_sizes=grp_sizes, col_grp_sizes=None,
                    row_grp_names=grp_names, col_grp_names=None,
                    vmin=0, vmax=1.0, resize='up', diagonal=False)

#%%
vmin=0
vmax=0.1
plot_pairwise_matrix(np.abs(influ_ij), title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes, col_grp_sizes=grp_sizes,
                    row_grp_names=grp_names, col_grp_names=grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)
plot_pairwise_matrix(influ_ij, title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes, col_grp_sizes=grp_sizes,
                    row_grp_names=grp_names, col_grp_names=grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)
plot_pairwise_matrix(-influ_ij, title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes, col_grp_sizes=grp_sizes,
                    row_grp_names=grp_names, col_grp_names=grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)

#%% plot the histogram of the influence_ij with matplotlib

plt.hist(influ_ij.flatten(), bins=200, range=(-0.1, 0.1))
# show the histogram of only the positive values
# plt.hist(-influ_ij[influ_ij > 0].flatten(), bins=100, range=(-0.2, 0.))
plt.hist(-influ_ij.flatten(), bins=200, range=(-0.1, 0.1))

plt.xlabel('Influence')
plt.ylabel('Count')
# set ylimits to 0, 10000
plt.ylim(0, 2e5)
plt.title('Histogram of Influence')
plt.show()

#%% plot the histogram of the influence_ij with matplotlib
plt.hist(meta_df['mem_score'].values, bins=201, range=(-0.05, 0.3))
plt.xlabel('Memorization Score')
plt.ylabel('Count')
plt.title('Histogram of Influence')
plt.show()
#%% Plot influence_ij for samples with tr_avg > 0.92
iloc_highF1 = np.where(meta_df['tr_avg'] > 0.92)[0]

meta_df_highF1 = meta_df.iloc[iloc_highF1]

meta_df_highF1, grp_sizes_highF1, grp_names_highF1 = collate_sort_df(meta_df_highF1)
plot_pairwise_matrix(influ_ij[iloc_highF1][:, iloc_highF1], title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes_highF1, col_grp_sizes=grp_sizes_highF1,
                    row_grp_names=grp_names_highF1, col_grp_names=grp_names_highF1,
                    vmin=0, vmax=0.1, resize='up', diagonal=False, average=False)

#%%
plot_pairwise_matrix(-influ_ij[iloc_highF1][:, iloc_highF1], title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=grp_sizes_highF1, col_grp_sizes=grp_sizes_highF1,
                    row_grp_names=grp_names_highF1, col_grp_names=grp_names_highF1,
                    vmin=0, vmax=0.1, resize='up', diagonal=False, average=False)
#%% Plot tr_avg vs vl_avg
glance_midat.violin_df_groupby(meta_df, split_input=meta_df, groupby='moltype',
                               y='tr_avg', split_y='vl_avg',
                               title='Distribution of TR and VL F1 scores by moltype',
                               xlabel='Moltype', ylabel='F1 score',
                               title_font_size=23, save_prefix='rand50_meta', 
                               yrange=[0, 1.16], img_height=600, img_width=1400)

#%% Plot meta_df with
glance_midat.violin_df_groupby(meta_df, split_input=meta_df, 
                               groupby='moltype', yrange=(0, 0.6),
                               y='mem_score_normalized', split_y='mem_score',
                               title='Distribution of memorization scores by moltype',
                               xlabel='Moltype', ylabel='Mem. score',
                               title_font_size=23,
                               save_prefix='rand50_meta_all', img_height=600, img_width=1400)
#%% Plot meta_df with tr_avg > 0.8
glance_midat.violin_df_groupby(meta_df[meta_df['tr_avg'] > 0.97], 
                               split_input=meta_df[meta_df['tr_avg'] > 0.97], 
                               groupby='moltype', yrange=(0, 0.6),
                               y='mem_score_normalized', split_y='mem_score',
                               title='Distribution of memorization scores by moltype',
                               xlabel='Moltype', ylabel='Mem. score',
                               title_font_size=23,
                               save_prefix='rand50_meta_highF1', img_height=600, img_width=1400)

#%% scatter plot of mem_score vs. tr_avg
# plt.scatter(meta_df['tr_avg'], meta_df['mem_score'], s=1)
plt.scatter(meta_df['tr_avg'], meta_df['vl_avg'], s=1)
# plot a vertical line at tr_avg = 92
plt.axvline(x=0.92, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=0.97, color='r', linestyle='--', linewidth=0.5)
# label the vertical lines
plt.text(0.90, 0.1, '0.92', color='r', fontsize=8)
plt.text(0.95, 0.05, '0.97', color='r', fontsize=8)
plt.xlabel('tr_avg')
plt.ylabel('mem_score')
plt.title('Memorization score vs. tr_avg')
# plt.xscale('log')
# plt.yscale('log')
plt.show()
#%% display the row in meta_df with the largest mem_score
max_idx = meta_df['mem_score'].idxmax()
print(f'Max mem_score: {meta_df["mem_score"].max()}')


#%% Define the function to collect seq2ct_loo metrics
def collect_loo_metrics(
        folder_path=os.path.join(os.path.expanduser('~'), 'bench/contarna/seq2ct_loo'),
        metric='f1',
        upsample=False,
        model_cutoff=0.85):

    folder_path=os.path.join(os.path.expanduser('~'), 'bench/contarna/seq2ct_loo')
    metric='f1'
    upsample=False

    ts_files = glob.glob(os.path.join(folder_path, 'strive.libset_len30-600_nr80_train-valid.l4c64.validobj.loo*/*_test.eval/eval_loss_meta.csv'), recursive=True)
    if upsample:
        ts_files = [_s for _s in ts_files if 'upsample' in _s]
    else:
        ts_files = [_s for _s in ts_files if 'upsample' not in _s]
    print(f'Number of test files (upsample={upsample}): {len(ts_files)}')

    # load the first VL file and its corresponding TR file
    ts_df0 = pd.read_csv(ts_files[0])
    tr_df0 = pd.read_csv(ts_files[0].replace('_test.eval', '_train-valid.eval'))
        
    # group by "moltype" and get the idx list for each group

    tr_df0, tr_grp_sizes, tr_grp_names = collate_sort_df(tr_df0)
    tr_df0.set_index('idx', drop=False, inplace=True)
    tr_df0.index.name = 'index'
    idx2irow = get_dfcol2irow(tr_df0, key='idx')

    ts_df0, ts_grp_sizes, ts_grp_names = collate_sort_df(ts_df0)
    ts_df0.set_index('idx', drop=False, inplace=True)
    ts_df0.index.name = 'index'
    idx2icol = get_dfcol2irow(ts_df0, key='idx')

    ts_metrics = np.zeros((len(tr_df0), len(ts_df0)))
    tr_metrics = np.zeros((len(tr_df0), len(tr_df0)))

    for i, vl_csv in enumerate(ts_files):
        ts_df = pd.read_csv(vl_csv)

        idx_loo = int(vl_csv.split('/')[-3].split('.')[-1][3:])
        irow2mat = idx2irow[idx_loo]
        icol2mat = idx2icol[ts_df['idx'].values]
        ts_metrics[irow2mat][icol2mat] = ts_df[metric].values
        
        tr_csv = vl_csv.replace('_test.eval', '_train-valid.eval')
        tr_df = pd.read_csv(tr_csv)

        icol2mat = idx2irow[tr_df['idx'].values]
        tr_metrics[irow2mat][icol2mat] = tr_df[metric].values

    if model_cutoff is not None:
        # remove the models with mean f1 score < model_cutoff
        tr_mean_per_model = np.mean(tr_metrics, axis=1)
        # find index of models with mean f1 score < model_cutoff
        cutoff_idx = np.where(tr_mean_per_model <= model_cutoff)[0]
        if len(cutoff_idx) > 0:
            print(f'Removing {len(cutoff_idx)} models with mean f1 score < {model_cutoff}')
            tr_metrics = np.delete(tr_metrics, cutoff_idx, axis=0)
            tr_metrics = np.delete(tr_metrics, cutoff_idx, axis=1)
            ts_metrics = np.delete(ts_metrics, cutoff_idx, axis=0)
            ts_files = [ts_files[i] for i in range(len(ts_files)) if i not in cutoff_idx]
            # get the moltype of the models to be removed
            cutoff_moltype = tr_df0.iloc[cutoff_idx]['moltype'].value_counts()
            for igrp, grp_name in enumerate(tr_grp_names):
                if grp_name in cutoff_moltype.index:
                    print(f'Removing {cutoff_moltype[grp_name]} models from group {grp_name}')
                    tr_grp_sizes[igrp] -= cutoff_moltype[grp_name]
            
    ts_values_sum = np.sum(ts_metrics, axis=0, keepdims=True)
    influ_tr2ts = (ts_values_sum - ts_metrics) / (ts_metrics.shape[0] - 1) - ts_metrics

    # apply the same operation to tr_values
    tr_values_sum = np.sum(tr_metrics, axis=0, keepdims=True)
    influ_tr2tr = (tr_values_sum - tr_metrics) / (tr_metrics.shape[0] - 1) - tr_metrics

    return ts_metrics, tr_metrics, influ_tr2ts, influ_tr2tr, tr_grp_sizes, ts_grp_sizes, tr_grp_names, ts_grp_names

#%% Load seq2ct_loo metrics
ts_metrics, tr_metrics, influ_tr2ts, influ_tr2tr, \
    tr_grp_sizes, ts_grp_sizes, tr_grp_names, ts_grp_names = collect_loo_metrics()

#%% Plot tr_metrics
plot_pairwise_matrix(tr_metrics, title='tr_values', xlabel='Train', ylabel='LOO Model', cmap='gray',
                    row_grp_sizes=tr_grp_sizes, col_grp_sizes=tr_grp_sizes,
                    row_grp_names=tr_grp_names, col_grp_names=tr_grp_names,
                    vmin=0, vmax=1, sizeup=True,)
#%% Plot ts_metrics
plot_pairwise_matrix(ts_metrics, title=None, xlabel='Test', ylabel='LOO Model', cmap='gray',
                    row_grp_sizes=tr_grp_sizes, col_grp_sizes=ts_grp_sizes,
                    row_grp_names=tr_grp_names, col_grp_names=ts_grp_names,
                    vmin=0, vmax=0.8, sizeup=True,)

#%% Plot influence_tr_ts
plot_pairwise_matrix(influ_tr2ts, title='influence_tr_ts', xlabel='idx', ylabel='idx', cmap='gray',
                    row_grp_sizes=tr_grp_sizes, col_grp_sizes=ts_grp_sizes,
                    row_grp_names=tr_grp_names, col_grp_names=ts_grp_names,
                    vmin=0, vmax=0.2, sizeup=True,)
#%% Plot influence_tr_ts with abs
plot_pairwise_matrix(np.abs(influ_tr2ts), title='influence_tr_ts', xlabel='idx', ylabel='idx', cmap='gray',
                    row_grp_sizes=tr_grp_sizes, col_grp_sizes=ts_grp_sizes,
                    row_grp_names=tr_grp_names, col_grp_names=ts_grp_names,
                    vmin=0., vmax=0.2, sizeup=True,)

#%% plot the histogram of the influence_ij with matplotlib
plt.hist(influ_tr2tr.flatten(), bins=201, range=(-0.2, 0.2))
plt.xlabel('Influence TR vs TR')
plt.ylabel('Count')
plt.title('Histogram of Influence')
plt.show()

plt.hist(influ_tr2ts.flatten(), bins=201, range=(-0.2, 0.2))
plt.xlabel('Influence TR vs TS')
plt.ylabel('Count')
plt.title('Histogram of Influence')
plt.show()

#%% plot the diagonal of influ_tr2tr with matplotlib
plt.hist(influ_tr2tr[np.diag_indices_from(influ_tr2tr)].flatten(), bins=201, range=(-0.2, 0.2))
plt.xlabel('Influence diagonal elements')
plt.ylabel('Count')
plt.title('Histogram of Influence diagonal elements (similar to mem_score)')
plt.show()
#%% plot the diagonal of influ_tr2ts with matplotlib


#%% Define load_pairwise_df()
# load the pairwise similarity matrix
from glance_midat import bake_pairwise_df

def load_pairwise_df(pkl_file='data/nr80-vs-nr80.rnadistance.alnIdentity_pairwise.pkl',
                     meta_file='data/libset_len30-600_nr80.pkl'):
    """ load the pairwise df from the pkl file"""
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            sim_df = pickle.load(f)
        print(f'Loaded {pkl_file}')
    else:
        raise FileNotFoundError(f'{pkl_file} not found')

    sim_mat, pairwise_ids, irows_with_NaNs = bake_pairwise_df(sim_df,
                square_dataframe=False,
                diagonal=1.0, 
                diagonal_nan=1.0,
                nan=None,
                conjugate_to_nan=True, 
                symmetric_nan=None,
                symmetrize=True,
                save_mat=False, 
                save_pkl=False)

    pairwise_idx = np.array([int(s.split('_')[0]) for s in pairwise_ids], dtype=int)
    # count the number of nans in parwise_mat
    num_nans = np.isnan(sim_mat).sum()
    print(f'Number of NaNs in pairwise_mat: {num_nans}')

    meta_df = pd.read_pickle(meta_file)

    # sort and get grp_sizes and grp_names from meta_df
    meta_df, grp_sizes, grp_names = collate_sort_df(meta_df)
    idx2i = np.empty(pairwise_idx.max() + 1, dtype=int)
    idx2i[meta_df['idx'].values] = np.arange(len(pairwise_idx))

    sim_mat[idx2i[pairwise_idx], :] = sim_mat
    sim_mat[:, idx2i[pairwise_idx]] = sim_mat

    return sim_mat, meta_df, grp_sizes, grp_names
#%%
# sim_mat, meta_df, grp_sizes, grp_names = load_pairwise_df()

sim_mat, meta_df, grp_sizes, grp_names = load_pairwise_df(
    pkl_file='/home/xqiu/database/contarna/metafam2d_2023/v3.0/metafam2d.rnaforester.seqIdentity_pairwise.pkl',
    meta_file='/home/xqiu/database/contarna/metafam2d_2023/v3.0/metafam2d.pkl',)

#%%
plot_pairwise_matrix(sim_mat, title='pairwise_mat', xlabel='idx', ylabel='idx', cmap='inferno',
                    row_grp_sizes=grp_sizes, col_grp_sizes=grp_sizes,
                    row_grp_names=grp_names, col_grp_names=grp_names,
                    vmin=0.1, vmax=0.6, resize=False,
                    diagonal=False, average=False,
                    )
#%%
sim_mat_resized, grp_sizes_resized = resize_matrix_grps(sim_mat, grp_sizes, method=100, dims=[0, 1])
plot_pairwise_matrix(sim_mat_resized, title='pairwise_mat', xlabel='idx', ylabel='idx', cmap='inferno',
                    row_grp_sizes=grp_sizes_resized, col_grp_sizes=grp_sizes_resized,
                    row_grp_names=grp_names, col_grp_names=grp_names,
                    vmin=0.1, vmax=0.95, resize=False,
                    diagonal=False, average=False,
                    )
#%%
# # scatter plot of influence_ij vs. sim_ij with matplotlib
# import matplotlib.pyplot as plt
# plt.scatter(np.tril(sim_ij).flatten(), np.tril(influence_ij).flatten(), s=1)
# plt.xlabel('sim_ij')
# plt.ylabel('influence_ij')
# plt.show()

sim_triu = sim_ij[np.triu_indices(sim_ij.shape[0], k=1)]
influence_triu = influ_ij[np.triu_indices(influ_ij.shape[0], k=1)]
# scatter plot of influence_triu vs. sim_triu
plt.scatter(sim_triu, influence_triu, s=1)
plt.xlabel('sim_ij')
plt.ylabel('influence_ij')
plt.show()

#%%
# sample 10000 pairs of samples and plot again
import random
idx_pairs = random.sample(range(sim_triu.shape[0]), 10000)
plt.scatter(sim_triu[idx_pairs], influence_triu[idx_pairs], s=1)
plt.xlabel('sim_ij')
plt.ylabel('influence_ij')
plt.show()

#%%
import matplotlib.pyplot as plt

# Create a figure with multiple subplots

#------------------- Generating a Kernal Density Dataframe from the pickle data---------------------
# input the pickle package into the function to pull out the kde scores based on the pickle data
def kde_scores(dataframe):
    def kde(row):
        row = 1 - row  # Invert the row values
        sigma = np.std(row)
        mu = np.mean(row)
        f = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((row - mu) ** 2) / sigma ** 2)
        return f

    kde_scores = {}
    for idx, row in dataframe.iterrows(): # Iterate through each row in the DataFrame and compute the mean KDE score
        row_values = row.values
        kde_values = kde(row_values)  # Vectorized KDE for the entire row
        kde_scores[idx] = np.mean(kde_values)  # Compute the mean KDE score for the row

    kde_scores_df = pd.DataFrame(list(kde_scores.items()), columns=['idx', 'similarity_score']) # Convert the results to a DataFrame
    return kde_scores_df

#-------------------Plotter Function for any two dataframes---------------------
def exp_decay(x, a, b, c):# Exponential decay function
        return a * np.exp(-b * x) + c

def plotter(memscoredf, simscoredf, simscore_name='Data Frame Name'):
    # Ensuring 'idx' columns are the same data type
    memscoredf['idx'] = memscoredf['idx'].astype(str)
    simscoredf['idx'] = simscoredf['idx'].astype(str)
    
    # Merging DataFrames on 'idx'
    merged_df = pd.merge(memscoredf, simscoredf, on='idx', how='inner')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])

    # Fitting data
    x = merged_df['similarity_score']
    y = merged_df['Memorization_Score']
    initial_guess = [1, 1, 1]
    params, covariance = curve_fit(exp_decay, x, y, p0=initial_guess)
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = exp_decay(x_fit, *params)

    # Unique moltypes and color mapping
    moltypes = merged_df['moltype'].unique()
    color_map = {moltype: color for moltype, color in zip(moltypes, plt.cm.tab10.colors[:len(moltypes)])}

    # Setting up the figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Smaller figure size

    # Plot 1: Memorization Score vs Distribution Score (KDE by Moltype)
    for moltype in moltypes:
        subset = merged_df[merged_df['moltype'] == moltype]
        axs[0].scatter(subset['similarity_score'], subset['Memorization_Score'],
                       color=color_map[moltype], label=f'Moltype {moltype}')
    axs[0].plot(x_fit, y_fit, color='red', label=f'Exp. Decay fit: {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[0].set_xlabel('Distribution Score')
    axs[0].set_ylabel('Memorization Score')
    axs[0].set_yscale('log')
    axs[0].set_title(f'Memorization Score vs Distribution Score ({simscore_name} by Moltype)')
    axs[0].legend()

    # Plot 2: Best Fit Line with Error Bars
    perr = np.sqrt(np.diag(covariance))  # Standard deviation of parameters
    y_err = exp_decay(x_fit, *(params + perr)) - exp_decay(x_fit, *params)
    axs[1].plot(x_fit, y_fit, color='red', label=f'Best Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[1].fill_between(x_fit, y_fit - y_err, y_fit + y_err, color='gray', alpha=0.3, label='Fit Uncertainty')
    axs[1].set_xlabel('Similarity Score')
    axs[1].set_ylabel('Memorization Score')
    axs[1].set_yscale('log')
    axs[1].set_title(f'Memorization Score vs Similarity Score ({simscore_name} with Uncertainty)')
    axs[1].legend()

    # Plot 3: Bubble Chart to Show Density of Points
    density, x_edges, y_edges = np.histogram2d(x, y, bins=50)
    x_center = (x_edges[:-1] + x_edges[1:]) / 2
    y_center = (y_edges[:-1] + y_edges[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_center, y_center)
    axs[2].scatter(x_mesh.ravel(), y_mesh.ravel(), s=density.ravel() * 10, alpha=0.6, label='Density of Points')
    axs[2].plot(x_fit, y_fit, color='red', label=f'Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[2].set_xlabel('Similarity Score')
    axs[2].set_ylabel('Memorization Score')
    axs[2].set_title(f'Memorization Score vs Similarity Score ({simscore_name} Bubble Chart)')
    axs[2].legend()

    # Adjusting layout for better spacing
    plt.tight_layout()
    plt.show()

    print(f'The exponential decay equation is: y = {params[0]:.2f} * e^(-{params[1]:.2f} * x) + {params[2]:.2f}')


#-------------------Exponential Decay Function for the pickle data--------------------
def exponential_sim_scores(pickle_df):
    # Constants for the exponential decay function
    A = 10  # Maximum points for the highest score (close to 1)
    B = np.log(A) / (1 - 0.85)  # Solve for B so that a score of 0.85 maps to 1 point

    def map_similarity_to_points(score):
        if pd.isna(score) or score >= 1 or score < 0.0:  # Exclude NaN and out-of-bound values
            return 0
        return A * np.exp(-B * (1 - score))  # Apply exponential decay function
    # Exponetial decay is fit by the equation y = 6.66e^-(1-x)

    similarity_scores = {} # Calculate similarity scores for each row
    for idx, row in pickle_df.iterrows():  # Iterate over each row by index
        points = row.apply(map_similarity_to_points).sum()  # Sum points in the row, ignoring NaN
        similarity_scores[idx] = points  # Store the total points for each idx
    similarity_scores_df = pd.DataFrame(list(similarity_scores.items()), columns=['idx', 'similarity_score'])# Convert the results to a DataFrame
    return similarity_scores_df


#-------------------- Quanta model of Simmilarity Score for pickle data ----------------------
def quanta_simscore(pickle):
    def quanta_simscore(score): # Function to map similarity score to points based on the new rules
        if score > 0.4:  #Only add a point if the simmialrity score is greater than 0.7
            return 1

    similarity_scores = {} # Empty dictionary 
    for idx, row in pickle.iterrows(): # Iterate over each row
        points = row.apply(quanta_simscore).sum() # Apply the mapping function to each similarity score in the row and sum the points, ignoring NaN
        similarity_scores[idx] = points # Store the total points as the overall similarity score for each idx

    quanta_simscore_df = pd.DataFrame(list(similarity_scores.items()), columns=['idx', 'similarity_score']) # Convert the results to a DataFrame
    return quanta_simscore_df

#----------------------Functions to sort for only high performing F1 scores (upsample)-----------------------
def f1above(df, threshold):
    df = df[df['tr_avg_f1'] > threshold] #Return only rows where the TR F1 score is above the threshold
    return df

def f1_above_median_per_moltype(df):
    median_f1 = df.groupby('moltype')['tr_avg_f1'].transform('median') # Calculate the median F1 score for each moltype
    df = df[df['tr_avg_f1'] > median_f1] # Filter rows where the TR F1 score is above the median
    return df

#-----------------------Plotly violin plotter -------------------------------
def plot_f1_density(df):
    fig = go.Figure()  # Initialize plotly figure
    moltypes = df['moltype'].unique()  # Get unique moltypes
    # Loop through each moltype to create the density plot
    for moltype in moltypes:
        subset = df[df['moltype'] == moltype]
        # Create violin plot for tr_avg_f1
        fig.add_trace(go.Violin(
            x=[moltype] * len(subset),  # Group by moltype
            y=subset['tr_avg_f1'],
            side='negative',  # Place the training F1 scores on the left
            line_color='blue',
            name=f"{moltype} tr_avg_f1",
            points=False,
            showlegend=False  ))

        # Create violin plot for vl_avg_f1
        fig.add_trace(go.Violin(
            x=[moltype] * len(subset),
            y=subset['vl_avg_f1'],
            side='positive',  # Place the validation F1 scores on the right
            line_color='green',
            name=f"{moltype} vl_avg_f1",
            points=False,
            showlegend=False ))
        # Add dashed horizontal lines for medians
        median_tr = subset['tr_avg_f1'].median()
        median_vl = subset['vl_avg_f1'].median()
        fig.add_shape(
            type="line",
            x0=moltypes.tolist().index(moltype) - 0.4,  # Align with moltype category
            x1=moltypes.tolist().index(moltype) + 0.4,
            y0=median_tr,
            y1=median_tr,
            line=dict(color="blue", width=2, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=moltypes.tolist().index(moltype) - 0.4,
            x1=moltypes.tolist().index(moltype) + 0.4,
            y0=median_vl,
            y1=median_vl,
            line=dict(color="green", width=2, dash="dash"),
        )

    fig.update_layout(
        title="Violin Density Plot for F1 Scores with Medians",
        xaxis=dict(title="Moltype", tickvals=list(range(len(moltypes))), ticktext=moltypes),
        yaxis_title="F1 Score",
        violingap=0.3,
        violinmode='overlay',
        showlegend=False,)
    fig.show()


#Input the dfs that contain the memorization scores
def overlay_upsample_normal(normal_df, upsample_df):
    normal = f1_above_median_per_moltype(normal_df) # Extract high-performing F1 scores
    upsampled = f1_above_median_per_moltype(upsample_df)
    similarity = exponential_sim_scores(pickle) # Calculate similarity scores for the pickle data

    normal.loc[:, 'idx'] = normal['idx'].astype(str)
    upsampled.loc[:, 'idx'] = upsampled['idx'].astype(str)
    similarity.loc[:, 'idx'] = similarity['idx'].astype(str)
    
    # Merging DataFrames on 'idx'
    merged_normal = pd.merge(normal, similarity, on='idx', how='inner')
    merged_normal = merged_normal.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])
    merged_upsampled = pd.merge(upsampled, similarity, on='idx', how='inner')
    merged_upsampled = merged_upsampled.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])

    x_normal = merged_normal['similarity_score']
    y_normal = merged_normal['Memorization_Score']
    x_upsampled = merged_upsampled['similarity_score']
    y_upsampled = merged_upsampled['Memorization_Score']
    
    # Fit exponential decay 
    initial_guess = [1, 1, 1]
    params_normal, covariance_normal = curve_fit(exp_decay, x_normal, y_normal, p0=initial_guess)
    x_fit_normal = np.linspace(min(x_normal), max(x_normal), 500)
    y_fit_normal = exp_decay(x_fit_normal, *params_normal)
    params_upsampled, covariance_upsampled = curve_fit(exp_decay, x_upsampled, y_upsampled, p0=initial_guess)
    x_fit_upsampled = np.linspace(min(x_upsampled), max(x_upsampled), 500)
    y_fit_upsampled = exp_decay(x_fit_upsampled, *params_upsampled)
    
    # Print the equations for the fits
    print(f"Normal Data Fit: y = {params_normal[0]:.5f} * exp(-{params_normal[1]:.5f} * x) + {params_normal[2]:.5f}")
    print(f"Upsampled Data Fit: y = {params_upsampled[0]:.5f} * exp(-{params_upsampled[1]:.5f} * x) + {params_upsampled[2]:.5f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_normal, y_normal, color='blue', alpha=0.5, label='Normal Data')
    plt.plot(x_fit_normal, y_fit_normal, color='blue', linestyle='--', label='Normal Fit')
    plt.scatter(x_upsampled, y_upsampled, color='orange', alpha=0.5, label='Upsampled Data')
    plt.plot(x_fit_upsampled, y_fit_upsampled, color='orange', linestyle='--', label='Upsampled Fit')
    plt.xlabel('Similarity Score')
    plt.ylabel('Memorization Score')
    plt.yscale('log')
    plt.title('Overlay of Normal and Upsampled Data with Exponential Fits')
    plt.legend()
    plt.show()



#------------------------------Influnce Score Generator and Plotter-----------------------------

def influence_matrix(directory):
    all_contributions = {}  # Dictionary to store contributions
    file_metadata = []  # List of file path and type
    train_file_data = {}  # Dictionary to store training file contents 
    for root, _, files in os.walk(directory): # Collect file paths for processing
        for file_name in files:
            if file_name.endswith(".csv") and file_name.startswith("eval_loss_meta"):
                split_type = "train" if root.endswith("_TR.eval") else "validation" if root.endswith("_VL.eval") else None #loading files in and assigning as TR or VL
                if split_type:
                    file_metadata.append((os.path.join(root, file_name), split_type)) #adding to storage

    # Load validation files first. This only looks at the validation files not the training files yet
    for file_path, split_type in file_metadata:
        if split_type != "validation":
            continue  # Skip training files here
        try:
            data = pd.read_csv(file_path, usecols=['idx', 'f1']) # Load only necessary columns of idx and f1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        avg_f1 = data['f1'].mean() #average f1 score for the file across all idx valued
        contributions = data.assign(contribution=data['f1'] - avg_f1).groupby('idx')['contribution'].apply(list) #finds the difference between the f1 score for each point and the average of that file
        for idx, contrib_list in contributions.items(): #storing the contribution of each data point to the dictionary
            all_contributions.setdefault(idx, []).extend(contrib_list)
    unique_datapoints = list(all_contributions.keys()) # list of all datapoints for future plotting
    num_points = len(unique_datapoints) # Number of unique data points
    mean_contributions = {dp: np.mean(contribs) for dp, contribs in all_contributions.items()} # Precompute mean contributions
    # Load all training files into memory for fast lookup
    for file_path, split_type in file_metadata:
        if split_type == "train":
            try:
                data = pd.read_csv(file_path, usecols=['idx'])
                train_file_data[file_path] = set(data['idx'].values)  # Store set of indices for lookup
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    pairwise_matrix = np.zeros((num_points, num_points)) # Initialize pairwise matrix

    for i, dp1 in enumerate(unique_datapoints): #iterates over all datapoints to calculate influnce
        for j, dp2 in enumerate(unique_datapoints):
            if i == j:
                continue  # Skip point if trying to find influnce on self

            total_contributions = []

            # Check each file in file_metadata
            for file_path, split_type in file_metadata: # For influnce of i on j, skip files where j is in training set
                if split_type == "train" and dp2 in train_file_data[file_path]:
                    continue  # Skip this training file if dp2 is in it
                # Compute influence only using allowed validation and training files
                if dp1 in mean_contributions and dp2 in mean_contributions:
                    total_contributions.append(mean_contributions[dp1] - mean_contributions[dp2]) #Finds difference in mean contributions between i and j

            # Compute mean influence or assign 0 if no contributions exist
            pairwise_matrix[i, j] = np.mean(total_contributions) if total_contributions else 0 #storing data or assigning value of zero

    pairwise_df = pd.DataFrame(pairwise_matrix, index=unique_datapoints, columns=unique_datapoints) #making matrix into a dataframe
    return pairwise_df


def plot_influence_vs_similarity(pairwise_influence_df):
    similarity_df= pickle
    # Convert indices and column names to strings, then extract the portion before the underscore
    pairwise_influence_df.index = pairwise_influence_df.index.astype(str).str.split('_').str[0]
    pairwise_influence_df.columns = pairwise_influence_df.columns.astype(str).str.split('_').str[0]
    similarity_df.index = similarity_df.index.astype(str).str.split('_').str[0]
    similarity_df.columns = similarity_df.columns.astype(str).str.split('_').str[0]
    # Ensure indices and columns match in both DataFrames
    matching_indices = pairwise_influence_df.index.intersection(similarity_df.index)
    matching_columns = pairwise_influence_df.columns.intersection(similarity_df.columns)
    similarity_df = similarity_df.loc[matching_indices, matching_columns]  # Filter both matrices to matching rows & columns
    pairwise_influence_df = pairwise_influence_df.loc[matching_indices, matching_columns]
    influence_scores = []
    similarity_scores = []
    seen_pairs = set()  # Track (idx, col) pairs to avoid duplicates

    for idx in matching_indices:
        for col in matching_columns:
            if idx in similarity_df.index and col in similarity_df.columns:
                # Ensure we only process (idx, col) when idx < col (avoids swapping duplicates)
                if idx < col:
                    similarity_score = similarity_df.loc[idx, col]
                    influence_score = pairwise_influence_df.loc[idx, col]

                    similarity_score = pd.to_numeric(similarity_score, errors='coerce')  # Convert to number, NaN if invalid
                    influence_score = pd.to_numeric(influence_score, errors='coerce')

                    # Avoid NaN values and ensure each pair is unique
                    if pd.notna(similarity_score) and pd.notna(influence_score):
                        pair_key = tuple(sorted((idx, col)))  # Create a sorted tuple to ensure order consistency
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            similarity_scores.append(similarity_score)
                            influence_scores.append(influence_score)

    if len(influence_scores) == 0 or len(similarity_scores) == 0:  # Ensure valid data
        print("No valid data for plotting. Check your DataFrames or filters.")
        return

    # Standard scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(influence_scores, similarity_scores, alpha=0.5)
    plt.title("Influence Score vs Similarity Score")
    plt.xlabel("Influence Score")
    plt.ylabel("Similarity Score")
    plt.grid(True)
    plt.show()



