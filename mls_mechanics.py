#%% Imoprt and define functions
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
import glance_mat
import glance_df
import brew_dfs
import importlib
importlib.reload(glance_midat)
importlib.reload(brew_dfs)

def plot_pairwise_matrix(matdata, title=None, xlabel=None, ylabel=None, figsize=None,
                         cmap='inferno', #'magma', # 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                         row_grp_sizes=None, col_grp_sizes=None,
                         row_grp_names=None, col_grp_names=None,
                         vmin=0, vmax=1, resize=False,
                         diagonal=True, hline=True, vline=True, average=True,
                         **kwargs):

    if figsize is not None:
        plt.figure(figsize=figsize)

    if resize:
        if row_grp_sizes is not None:
            matdata, row_grp_sizes = glance_mat.resize_matrix_by_group(matdata, row_grp_sizes, method=resize)
        if col_grp_sizes is not None:
            matdata, col_grp_sizes = glance_mat.resize_matrix_by_group(matdata.T, col_grp_sizes, method=resize)
            matdata = matdata.T

    plt.imshow(matdata, cmap=cmap, vmin=vmin, vmax=vmax)

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
                # plt.text(col_grp_iranges[i] + col_grp_sizes[i] / 2, -0.5, name, color='r', fontsize=8,
                #         ha='center', va='bottom', rotation=45)
                plt.text(col_grp_iranges[i] + col_grp_sizes[i] / 2, matdata.shape[0]+0.5, name, color='r', fontsize=8,
                        ha='center', va='top', rotation=45)
                
    # display the mean of each row and col group
    if average and row_grp_sizes is not None and col_grp_sizes is not None:
        for i in range(len(row_grp_sizes)):
            if row_grp_sizes[i] <= 0: continue
            for j in range(len(col_grp_sizes)):
                if col_grp_sizes[j] <= 0: continue
                block_data = matdata[row_grp_iranges[i]:row_grp_iranges[i+1], col_grp_iranges[j]:col_grp_iranges[j+1]]
                # get the mean of the block without NaN
                block_data = block_data[~np.isnan(block_data)]
                if block_data.size == 0: continue
                plt.text(col_grp_iranges[j] + col_grp_sizes[j] / 2,
                         row_grp_iranges[i] + row_grp_sizes[i] / 2,
                         f'{block_data.mean():.3f}', 
                         color=kwargs.get('average_color', 'w'), 
                         fontsize=kwargs.get('average_fontsize', 8),
                         ha='center', va='center', rotation=0)

    # draw a diagonal line from (0, 0) to (len(tr_df0), len(ts_df0))
    if diagonal:
        plt.plot([0, matdata.shape[1]-1], [0, matdata.shape[0]-1], color='r', linestyle='--', linewidth=0.5)

    # disable aspect ratio
    plt.gca().set_aspect('auto')
    plt.colorbar()
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    # only show the ticks for the first and last row/col
    plt.xticks([0, matdata.shape[1]-1], [0, matdata.shape[1]-1])
    plt.yticks([0, matdata.shape[0]-1], [0, matdata.shape[0]-1])
    plt.show()


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
    meta_df, grp_sizes, grp_names = brew_dfs.sort_df_by_groupsize(meta_df)
    meta_df = meta_df.set_index('idx', drop=False)
    meta_df.index.name = 'index'
    idx2iloc = brew_dfs.get_dfcol2iloc(meta_df, key='idx')

    # group by "moltype" and get the idx list for each group
    # idx_per_moltype ={}
    # for grp, df_grp in meta_df.groupby('moltype'):
    #     idx_per_moltype[grp] = df_grp['idx'].values

    tr_vl_metrics = np.zeros((num_samples, len(vl_files)))
    vl_isa = np.zeros((num_samples, len(vl_files)))
    for i, vl_csv in tqdm.tqdm(enumerate(vl_files)):
        vl_df = pd.read_csv(vl_csv)

        i2mat = idx2iloc[vl_df['idx'].values]
        tr_vl_metrics[i2mat, i] = vl_df[metric].values
        vl_isa[i2mat, i] = 1 # 1 if in validation set, 0 if in training set
        
        tr_csv = vl_csv.replace('_VL.eval', '_TR.eval')
        tr_df = pd.read_csv(tr_csv)
        i2mat = idx2iloc[tr_df['idx'].values]
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

rand50_meta, rand50_metrics, ran50_tr_vl, rand50_influ, rand50_grp_sizes, rand50_grp_names = collect_rand50_metrics(model_cutoff=0.8)
rand50_meta.to_pickle('rand50_meta.pkl')

tr_metric_mean_per_model = np.sum(rand50_metrics * (1 - ran50_tr_vl), axis=0) / np.sum(1 - ran50_tr_vl, axis=0)
vl_metric_mean_per_model = np.sum(rand50_metrics * ran50_tr_vl, axis=0) / np.sum(ran50_tr_vl, axis=0)
print(f'Mean TR metric per model: {tr_metric_mean_per_model.mean():.3f} +/- {tr_metric_mean_per_model.std():.3f}')
print(f'Mean VL metric per model: {vl_metric_mean_per_model.mean():.3f} +/- {vl_metric_mean_per_model.std():.3f}')
#%% Plot tr_metric_mean_per_model vs vl_metric_mean_per_model with matplotlib
plt.scatter(tr_metric_mean_per_model, vl_metric_mean_per_model, s=30)
plt.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=0.5)
plt.xlim(0.87, 0.94)
plt.ylim(0.8, 0.88)
plt.xlabel('TR metric mean per model')
plt.ylabel('VL metric mean per model')
plt.title('TR vs VL metric mean per model')
plt.show()

#%% Plot tr_avg vs vl_avg with matplotlib
plt.scatter(rand50_meta['tr_avg'], rand50_meta['vl_avg'], s=1)
plt.axvline(x=0.92, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=0.97, color='r', linestyle='--', linewidth=0.5)
plt.text(0.90, 0.51, '0.92', color='r', fontsize=8)
plt.text(0.95, 0.55, '0.97', color='r', fontsize=8)
plt.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=0.5)
plt.xlabel('tr_avg')
plt.ylabel('vl_avg')
plt.title('VL vs TR avg')
plt.xlim(0., 1.02)
plt.ylim(0., 1.02)
plt.show()

#%% Plot tr_vl_metrics
plot_pairwise_matrix(rand50_metrics, title='TR and VL F1 values of Rand50 Models', xlabel='Model #', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=rand50_grp_sizes, col_grp_sizes=None,
                    row_grp_names=rand50_grp_names, col_grp_names=None,
                    vmin=0, vmax=1.0, resize='up', diagonal=False,)

#%% Plot vl_isa
plot_pairwise_matrix(ran50_tr_vl, title='TR or VL flags of Rand50 Models', xlabel='Model #', ylabel='Train-Valid', cmap='gray',
                    row_grp_sizes=rand50_grp_sizes, col_grp_sizes=None,
                    row_grp_names=rand50_grp_names, col_grp_names=None,
                    vmin=0, vmax=1.0, resize='up', diagonal=False)

#%% Plot tr_avg vs vl_avg with glance_df
rand50_gdf = glance_df.MyDataFrame(rand50_meta)
rand50_gdf.plx_xys(fmt='scatter', x='tr_avg', ys='vl_avg', xtitle='TR_avg', ytitle='VL_avg', 
                    title='Average F1 values per sample from Rand50', lines=[[[0,0],[1,1]]],)

#%% Plot tr_avg vs vl_avg
glance_midat.violin_df_groupby(rand50_meta, split_input=rand50_meta, groupby='moltype',
                               y='tr_avg', split_y='vl_avg',
                               title='Distribution of TR and VL F1 scores by moltype',
                               xlabel='Moltype', ylabel='F1 score',
                               title_font_size=23, save_prefix='rand50_meta', 
                               yrange=[0, 1.16], img_height=600, img_width=1400)

#%% Plot meta_df with
glance_midat.violin_df_groupby(rand50_meta, split_input=rand50_meta, 
                               groupby='moltype', yrange=(0, 0.6),
                               y='mem_score_normalized', split_y='mem_score',
                               title='Distribution of memorization scores by moltype',
                               xlabel='Moltype', ylabel='Mem. score',
                               title_font_size=23,
                               save_prefix='rand50_meta_all', img_height=600, img_width=1400)
#%% Plot meta_df with tr_avg > 0.8
glance_midat.violin_df_groupby(rand50_meta[rand50_meta['tr_avg'] > 0.97], 
                               split_input=rand50_meta[rand50_meta['tr_avg'] > 0.97], 
                               groupby='moltype', yrange=(0, 0.6),
                               y='mem_score_normalized', split_y='mem_score',
                               title='Distribution of memorization scores by moltype',
                               xlabel='Moltype', ylabel='Mem. score',
                               title_font_size=23,
                               save_prefix='rand50_meta_highF1', img_height=600, img_width=1400)

#%% scatter plot of mem_score vs. tr_avg
# plt.scatter(meta_df['tr_avg'], meta_df['mem_score'], s=1)
plt.scatter(rand50_meta['tr_avg'], rand50_meta['mem_score'], s=1)
plt.axvline(x=0.92, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=0.97, color='r', linestyle='--', linewidth=0.5)
plt.text(0.90, 0.1, '0.92', color='r', fontsize=8)
plt.text(0.95, 0.05, '0.97', color='r', fontsize=8)
plt.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=0.5)

plt.xlabel('tr_avg')
plt.ylabel('mem_score')
plt.title('Memorization score vs. tr_avg')
# plt.xscale('log')
# plt.yscale('log')
plt.show()

#%% plot the histogram of the influence_ij with matplotlib
plt.hist(rand50_meta['mem_score'].values, bins=201, range=(-0.01, 0.5))
plt.xlabel('Memorization Score')
plt.ylabel('Count')
plt.title('Histogram of memorization scores')
plt.show()

#%% Plot influence_ij
vmin=0
vmax=0.1
plot_pairwise_matrix(np.abs(rand50_influ), title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=rand50_grp_sizes, col_grp_sizes=rand50_grp_sizes,
                    row_grp_names=rand50_grp_names, col_grp_names=rand50_grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)
plot_pairwise_matrix(rand50_influ, title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=rand50_grp_sizes, col_grp_sizes=rand50_grp_sizes,
                    row_grp_names=rand50_grp_names, col_grp_names=rand50_grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)
plot_pairwise_matrix(-rand50_influ, title=None, xlabel='Rand50 Model', ylabel='Train-Valid', cmap='inferno',
                    row_grp_sizes=rand50_grp_sizes, col_grp_sizes=rand50_grp_sizes,
                    row_grp_names=rand50_grp_names, col_grp_names=rand50_grp_names,
                    vmin=vmin, vmax=vmax, resize='up', diagonal=False, average=False,)

#%% plot the histogram of the influence_ij 
plt.hist(rand50_influ.flatten(), bins=200, range=(-0.1, 0.1))
plt.hist(-rand50_influ.flatten(), bins=200, range=(-0.1, 0.1))
plt.xlabel('Influence')
plt.ylabel('Count')
plt.ylim(0, 2e5)
plt.title('Histogram of Influence')
plt.show()
#%% Plot influence_ij for samples with tr_avg > 0.92
iloc_highF1 = np.where(rand50_meta['tr_avg'] > 0.92)[0]

meta_df_highF1 = rand50_meta.iloc[iloc_highF1]

meta_df_highF1, grp_sizes_highF1, grp_names_highF1 = brew_dfs.sort_df_by_groupsize(meta_df_highF1)
plot_pairwise_matrix(rand50_influ[iloc_highF1][:, iloc_highF1], title='Rand50 Model (F1 > 0.92)',
                     cmap='inferno',
                     row_grp_sizes=grp_sizes_highF1, col_grp_sizes=grp_sizes_highF1,
                     row_grp_names=grp_names_highF1, col_grp_names=grp_names_highF1,
                     vmin=0, vmax=0.1, resize='up', diagonal=False, average=False)

plot_pairwise_matrix(-rand50_influ[iloc_highF1][:, iloc_highF1], title='Rand50 Model (F1 > 0.92), Negative', 
                     cmap='inferno',
                     row_grp_sizes=grp_sizes_highF1, col_grp_sizes=grp_sizes_highF1,
                     row_grp_names=grp_names_highF1, col_grp_names=grp_names_highF1,
                     vmin=0, vmax=0.1, resize='up', diagonal=False, average=False)

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

    tr_df0, tr_grp_sizes, tr_grp_names = brew_dfs.sort_df_by_groupsize(tr_df0)
    tr_df0.set_index('idx', drop=False, inplace=True)
    tr_df0.index.name = 'index'
    idx2irow = brew_dfs.get_dfcol2iloc(tr_df0, key='idx')

    ts_df0, ts_grp_sizes, ts_grp_names = brew_dfs.sort_df_by_groupsize(ts_df0)
    ts_df0.set_index('idx', drop=False, inplace=True)
    ts_df0.index.name = 'index'
    idx2icol = brew_dfs.get_dfcol2iloc(ts_df0, key='idx')

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
                    vmin=0, vmax=1, resize='up', diagonal=False,)
#%% Plot ts_metrics
plot_pairwise_matrix(ts_metrics, title=None, xlabel='Test', ylabel='LOO Model', cmap='gray',
                    row_grp_sizes=tr_grp_sizes, col_grp_sizes=ts_grp_sizes,
                    row_grp_names=tr_grp_names, col_grp_names=ts_grp_names,
                    vmin=0, vmax=0.8, resize='up', diagonal=False)

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
plt.hist(-influ_tr2tr.flatten(), bins=201, range=(-0.2, 0.2))
plt.xlabel('Influence TR vs TR')
plt.ylabel('Count')
plt.title('Histogram of Influence')
plt.show()

plt.hist(influ_tr2ts.flatten(), bins=201, range=(-0.2, 0.2))
plt.hist(-influ_tr2ts.flatten(), bins=201, range=(-0.2, 0.2))

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


#%% Define load_pairwise_df()
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
    print(f'Loaded {meta_file}')

    # sort and get grp_sizes and grp_names from meta_df
    meta_df, grp_sizes, grp_names = brew_dfs.sort_df_by_groupsize(meta_df)
    # check if pairwise_idx the same as in meta_df['idx'].values
    if np.array_equal(np.sort(pairwise_idx), np.sort(meta_df['idx'].values)):
        print(f'Sorted pairwise_idx and meta_df["idx"].values are the same')
    else:
        raise ValueError(f'Sorted pairwise_idx and meta_df["idx"].values are not the same')
    
    idx2i = np.empty(pairwise_idx.max() + 1, dtype=int)
    idx2i[meta_df['idx'].values] = np.arange(len(pairwise_idx))

    pairwise_idx_aligned = np.empty_like(pairwise_idx)
    pairwise_idx_aligned[idx2i[pairwise_idx]] = pairwise_idx[:]

    # print(f'pairwise_idx: {pairwise_idx}')
    # print(f'meta_df["idx"].values: {meta_df["idx"].values}')
    if np.array_equal(pairwise_idx_aligned, meta_df['idx'].values):
        print(f'Aligned pairwise_idx and meta_df["idx"].values are the same')
    else:
        print(pairwise_idx_aligned - meta_df['idx'].values)
        raise ValueError(f'Aligned pairwise_idx and meta_df["idx"].values are not the same')

    sim_mat_aligned = np.empty_like(sim_mat)
    sim_mat_aligned[idx2i[pairwise_idx]] = sim_mat
    sim_mat[:, idx2i[pairwise_idx]] = sim_mat_aligned

    return sim_mat, meta_df, grp_sizes, grp_names
# %% Load the pairwise Sequence similarity
# sim_mat, meta_df, grp_sizes, grp_names = load_pairwise_df()

# sim_mat, meta_df, grp_sizes, grp_names = load_pairwise_df(
#     pkl_file='/home/xqiu/database/contarna/metafam2d_2023/v3.0/metafam2d.rnaforester.seqIdentity_pairwise.pkl',
#     meta_file='/home/xqiu/database/contarna/metafam2d_2023/v3.0/metafam2d.pkl',)

seqsim_mat, seqsim_df, seqsim_grp_sizes, seqsim_grp_names = load_pairwise_df(
    pkl_file='/home/xqiu/database/contarna/strive_2022/libset_len30-600_nr80_rmvpknots.rnaforester.seqIdentity_pairwise.pkl',
    meta_file='/home/xqiu/database/contarna/strive_2022/libset_len30-600_nr80.pkl',)

#%% plot the seqsim_mat
plot_pairwise_matrix(seqsim_mat, title='Sequence Similarity', xlabel=None, ylabel=None, cmap='inferno',
                    row_grp_sizes=seqsim_grp_sizes, col_grp_sizes=seqsim_grp_sizes,
                    row_grp_names=seqsim_grp_names, col_grp_names=seqsim_grp_names,
                    vmin=0.1, vmax=0.6, resize=100,
                    diagonal=False, average=True, average_color='w', average_fontsize=8,
                    )
#%% Define adjacency score function
def get_adj_score(sim_mat, sim_cutoff=0.5, weighting='exponential', exponent=4.2, gaussian_width=0.2, **kwargs):
    """
    Get the adjacency score for each samples in the seqsim_mat
    Args:
        sim_mat: the similarity matrix
        meta_df: the metadata dataframe
        sim_cutoff: the cutoff for the similarity score
        weighting: the weighting method, can be 'exponential' or 'linear' or 'gaussian'
        exponent: the exponent for the exponential weighting
        gaussian_width: the width for the gaussian weighting
        kwargs: other arguments for the weighting method
    Returns:
        adj_score: the adjacency score for each samples in the seqsim_mat
    """

    if weighting.upper() in ['EXPONENTIAL', 'EXP']:
        adj_score = sim_mat * np.exp(exponent * sim_mat)
    elif weighting.upper() in ['LINEAR', 'LIN']:
        adj_score = sim_mat
    elif weighting.upper() in ['GAUSSIAN', 'GAUSS']:
        adj_score = sim_mat * np.exp(-((sim_mat - 1) ** 2) / (2 * gaussian_width ** 2))
    else:
        raise ValueError(f'Unknown weighting method: {weighting}')
    
    # set the diagonal to 0
    np.fill_diagonal(adj_score, 0)

    if sim_cutoff is not None:
        mat_mask = (sim_mat >= sim_cutoff).astype(float)
    else:
        mat_mask = 1

    return np.nansum(adj_score * mat_mask, axis=1, )

#%% Get the adjacency score for each samples in the seqsim_mat
idx2iloc = brew_dfs.get_dfcol2iloc(seqsim_df, key='idx')
_rand50_seqsim = seqsim_mat[idx2iloc[rand50_meta['idx'].values]]
rand50_seqsim = _rand50_seqsim[:, idx2iloc[rand50_meta['idx'].values]]

#%% Plot adj_score vs mem_score in rand50_meta
rand50_meta['adj_score'] = get_adj_score(rand50_seqsim, sim_cutoff=0.4, weighting='exponential', exponent=4.2, gaussian_width=0.1) / 1000
# rand50_meta['adj_score'] = get_adj_score(rand50_seqsim, sim_cutoff=0.4, weighting='gaussian', exponent=4.2, gaussian_width=0.118) * 100

rand50_meta2plot = rand50_meta[rand50_meta['tr_avg'] > 0.8]
rand50_meta2plot = rand50_meta2plot[rand50_meta2plot['mem_score'] > 1e-4]
# plot each group with different color in a different subplot
fig, axs = plt.subplots(3, 3, figsize=(10, 8))
i = -1
for grp_name, df_grp in rand50_meta2plot.groupby('moltype'):
    i += 1
    ax = axs[i // 3, i % 3]
    ax.scatter(df_grp['adj_score'].values, df_grp['mem_score'].values, s=5, label=f'{grp_name} ({df_grp["mem_score"].mean():.2f})')
    # only show xlabel for hte bottom row
    if i // 3 == 2:
        ax.set_xlabel('Adjacency Score')
    if i % 3 == 0:
        ax.set_ylabel('Memorization Score')
    # ax.set_title(f'{grp_name}')
    # show text at the top right
    ax.set_yscale('log')
    ax.set_xlim(-0.01, 3)
    # ax.set_ylim(0.01, 1.1)
    # show legend at top right
    ax.legend(loc='upper right')

plt.show()


#%% plot the adjacency score
plt.hist(rand50_meta['adj_score'], bins=100)#, range=(adj_score.min(), adj_score.max()))
plt.xlabel('Adjacency Score')
plt.ylabel('Count')
plt.title('Histogram of Adjacency Score')
plt.show()

#%% Load the pairwise DBN similarity
dbnsim_mat, dbnsim_meta, dbnsim_grp_sizes, dbnsim_grp_names = load_pairwise_df(
    pkl_file='/home/xqiu/database/contarna/strive_2022/libset_len30-600_nr80_rmvpknots.rnaforester.dbnIdentity_pairwise.pkl',
    meta_file='/home/xqiu/database/contarna/strive_2022/libset_len30-600_nr80.pkl',)

plot_pairwise_matrix(dbnsim_mat, title='DBN Similarity', xlabel=None, ylabel=None, cmap='inferno',
                    row_grp_sizes=dbnsim_grp_sizes, col_grp_sizes=dbnsim_grp_sizes,
                    row_grp_names=dbnsim_grp_names, col_grp_names=dbnsim_grp_names,
                    vmin=0.1, vmax=0.9, resize=500,
                    diagonal=False, average=True, average_color='w', average_fontsize=8,
                    )


#%% Get the adjacency score for each samples in the seqsim_mat
idx2iloc = brew_dfs.get_dfcol2iloc(dbnsim_meta, key='idx')
_rand50_dbnsim = dbnsim_mat[idx2iloc[rand50_meta['idx'].values]]
rand50_dbnsim = _rand50_dbnsim[:, idx2iloc[rand50_meta['idx'].values]]

#%% Plot adj_score vs mem_score in rand50_meta
rand50_meta['adj_score'] = get_adj_score(rand50_dbnsim, sim_cutoff=0.4, weighting='exponential', exponent=4.2, gaussian_width=0.1) / 10000
# rand50_meta['adj_score'] = get_adj_score(rand50_seqsim, sim_cutoff=0.4, weighting='gaussian', exponent=4.2, gaussian_width=0.118) * 100

rand50_meta2plot = rand50_meta[rand50_meta['tr_avg'] > 0.8]
rand50_meta2plot = rand50_meta2plot[rand50_meta2plot['mem_score'] > 1e-4]
#%%
# Plot each group with different color in one panel
for grp_name, df_grp in rand50_meta2plot.groupby('moltype'):
    plt.scatter(df_grp['adj_score'].values, df_grp['mem_score'].values, s=5, label=f'{grp_name} ({df_grp["mem_score"].mean():.2f})')
plt.xlabel('DBN Adjacency Score')
plt.ylabel('Memorization Score')
plt.yscale('log')
plt.xlim(-0.01, 3.5)
plt.legend(loc='upper right')
plt.title('Adjacency Score vs Memorization Score')
plt.show()
#%%
# plot each group with different color in a different subplot
fig, axs = plt.subplots(3, 3, figsize=(10, 8))
i = -1
for grp_name, df_grp in rand50_meta2plot.groupby('moltype'):
    i += 1
    ax = axs[i // 3, i % 3]
    ax.scatter(df_grp['adj_score'].values, df_grp['mem_score'].values, s=5, label=f'{grp_name} ({df_grp["mem_score"].mean():.2f})')
    # only show xlabel for hte bottom row
    if i // 3 == 2:
        ax.set_xlabel('Adjacency Score')
    if i % 3 == 0:
        ax.set_ylabel('Memorization Score')
    # ax.set_title(f'{grp_name}')
    # show text at the top right
    ax.set_yscale('log')
    ax.set_xlim(-0.01, 3)
    # ax.set_ylim(0.01, 1.1)
    # show legend at top right
    ax.legend(loc='upper right')

plt.show()


#%% plot the adjacency score
plt.hist(rand50_meta['adj_score'], bins=100)#, range=(adj_score.min(), adj_score.max()))
plt.xlabel('Adjacency Score')
plt.ylabel('Count')
plt.title('Histogram of Adjacency Score')
plt.show()
