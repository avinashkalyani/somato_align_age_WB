#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os.path import join as pjoin
import numpy as np
import pandas as pd
import runpy
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# import functions from other scripts
file_globals = runpy.run_path('digit_classification_knn.py')
get_digit_indices = file_globals['get_digit_indices']
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


vox = 1001
age_idx = 0
k = 5
with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/projection_%s_%s_%s.npz' %(vox,age_idx,k), allow_pickle = True) as data:
    shared_all = data['proj_data']
    srms = data['srms']


# In[2]:


vox = 1001
age_idx = 0
k = 5
with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/COLUMN_projection_ALLVOX_80_1_10_BA3b.npz', allow_pickle = True) as data:
    shared_all = data['proj_data']
    srms = data['srms']


# In[7]:


proj_data_run1train = shared_all[0]
trained_srms_run1train = srms
proj_data_run2train = shared_all[1]
trained_srms_run2train =  srms


# In[8]:


# create finger-color arrays

# get ndarrays (nfingers, nvols)
digits_run1, digits_run2 = get_digit_indices()

# turn into flat arrays with values 1-5
digit_colors_run1, digit_colors_run2 = np.zeros(shape=(256)), np.zeros(shape=(256))
for finger_i in range(1,6):
    digit_colors_run1[digits_run1[finger_i-1]] = finger_i
    digit_colors_run2[digits_run2[finger_i-1]] = finger_i

# for some reason, the first entry in the first run's digit array is a 5, but should be a 1...
digit_colors_run1[0]=1.


# In[9]:


# make lists of data frames

# first run is training, second is test
projected_dfs_run2 = []
trained_srms_dfs_run1 = []

for sub_idx in range(proj_data_run1train.shape[0]):
    df = pd.DataFrame(proj_data_run1train[sub_idx].T)
    projected_dfs_run2.append(df)
    srm_df = pd.DataFrame(trained_srms_run1train[sub_idx].T)
    trained_srms_dfs_run1.append(srm_df)

# second run is training, first is test
projected_dfs_run1 = []
trained_srms_dfs_run2 = []

for sub_idx in range(proj_data_run2train.shape[0]):
    df = pd.DataFrame(proj_data_run2train[sub_idx].T)
    projected_dfs_run1.append(df)
    srm_df = pd.DataFrame(trained_srms_run2train[sub_idx].T)
    trained_srms_dfs_run2.append(srm_df)


# In[10]:




sns.set_style('whitegrid')


# In[11]:


# create finger-color arrays

# get ndarrays (nfingers, nvols)
digits_run1, digits_run2 = get_digit_indices()

# turn into flat arrays with values 1-5
digit_colors_run1, digit_colors_run2 = np.zeros(shape=(256)), np.zeros(shape=(256))
for finger_i in range(1,6):
    digit_colors_run1[digits_run1[finger_i-1]] = finger_i
    digit_colors_run2[digits_run2[finger_i-1]] = finger_i

# for some reason, the first entry in the first run's digit array is a 5, but should be a 1...
digit_colors_run1[0]=1.

# look at result to make sure it makes sens
plt.plot(digit_colors_run1, color='b')
plt.plot(digit_colors_run2, color='r')
plt.legend(['run1', 'run2'])
plt.title('finger color arrays')


# In[14]:


def plot_all_for_subject(sub_idx,
                         elevations_per_plot=(30,30,30,30),
                         reduce_method='PCA',
                         figsize=(8,5),
                         line_style='solid',
                         marker_style='.',
                         color_palette='muted',
                         fmt_style=None,
                         n_dimensions=3):
    
    print('plotting subject %i' % (sub_idx+1))
    
    run1_legend = ['d%i' %i for i in range(1,6)]
    run2_legend = ['d%i' %i for i in range(5,0,-1)]
    figkwargs = {'reduce':reduce_method, 'size':figsize, 
                 'linestyle':line_style, 'marker': marker_style, 
                 'palette':color_palette}  #, 'fmt':fmt_style } # , 'ndims':n_dimensions}
    
    # data from second run projected onto srm trained on first run
    geo = hyp.plot(projected_dfs_run2[sub_idx], title='projected data: subject %i, run 2.' % (sub_idx+1), 
                   group=digit_colors_run2, elev=elevations_per_plot[0], legend=run2_legend,
                   **figkwargs)
    #geo.fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/projection/proj_sub%i_run2'% (sub_idx+1), dpi=300)

    # srm trained on first run (with that subject held out)
    geo = hyp.plot(trained_srms_dfs_run1[sub_idx], title='SRM: subject %i held out, run 1.'% (sub_idx+1), 
                   group=digit_colors_run1, elev=elevations_per_plot[1], legend=run1_legend,
                   **figkwargs)
    #geo.fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/projection/srm_sub%i_run1'% (sub_idx+1), dpi=300)
    
    # data from first run projected onto srm trained on second run
    geo = hyp.plot(projected_dfs_run1[sub_idx], title='projected data: subject %i, run 1.'% (sub_idx+1), 
                   group=digit_colors_run1,  elev=elevations_per_plot[2], legend=run1_legend,
                   **figkwargs)
    #geo.fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/projection/proj_sub%i_run1'% (sub_idx+1), dpi=300)
    
    # srm trained on second run (with that subject held out)
    geo = hyp.plot(trained_srms_dfs_run2[sub_idx], title='SRM: subject %i held out, run 2.'% (sub_idx+1), 
                   group=digit_colors_run2, elev=elevations_per_plot[3], legend=run2_legend,
                   **figkwargs)
    #geo.fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/projection/srm_sub%i_run2'% (sub_idx+1), dpi=300)


# In[15]:


# plot the first few subjects
import hypertools as hyp
for subidx in range(19):
    plot_all_for_subject(subidx)  #, fmt_style='.', line_style=None, n_dimensions=2


# In[ ]:


# Different subjects from same run

# basic principle is [procrustes(source, target), target]
aligned_sub1and2_run2 = [hyp.tools.procrustes(projected_dfs_run2[1], projected_dfs_run2[0]), projected_dfs_run2[0]]
hyp.plot(aligned_sub1and2_run2, title='different subjects projected data\nfrom same run hyperaligned', 
         legend=['sub2', 'sub1'], reduce='PCA', size=[8,5])


# In[ ]:


# same subject's data from different runs

aligned_sub1_run1and2 = [hyp.tools.procrustes(projected_dfs_run2[0], projected_dfs_run1[0]), projected_dfs_run1[0]]
hyp.plot(aligned_sub1_run1and2, title="Subject 1's projected data\nfrom the two different runs hyperaligned", 
         legend=['run2', 'run1'], reduce='PCA', size=[8,5])


# In[ ]:


def plot_all_for_subject_shared_responses(sub_idx,
                         elevations_per_plot=(30,30,30,30),
                         figsize=(8,5),
                         line_style='solid',
                         marker_style='.',
                         color_palette='muted',
                         fmt_style=None,
                         n_dimensions=3):
    
    print('plotting subject %i' % (sub_idx+1))
    
    run1_legend = ['d%i' %i for i in range(1,6)]
    run2_legend = ['d%i' %i for i in range(5,0,-1)]
    figkwargs = {'reduce':None, 'size':figsize, 
                 'linestyle':line_style, 'marker': marker_style, 
                 'palette':color_palette}  #, 'fmt':fmt_style } # , 'ndims':n_dimensions}
    
    # data from second run projected onto srm trained on first run
    geo = hyp.plot(projected_dfs_run2[sub_idx][range(3)], title='projected data: subject %i, run 2.' % (sub_idx+1), 
                   group=digit_colors_run2, elev=elevations_per_plot[0], legend=run2_legend,
                   **figkwargs)

    # srm trained on first run (with that subject held out)
    geo = hyp.plot(trained_srms_dfs_run1[sub_idx][range(3)], title='SRM: subject %i held out, run 1.'% (sub_idx+1), 
                   group=digit_colors_run1, elev=elevations_per_plot[1], legend=run1_legend,
                   **figkwargs)
    
    # data from first run projected onto srm trained on second run
    geo = hyp.plot(projected_dfs_run1[sub_idx][range(3)], title='projected data: subject %i, run 1.'% (sub_idx+1), 
                   group=digit_colors_run1,  elev=elevations_per_plot[2], legend=run1_legend,
                   **figkwargs)
    
    # srm trained on second run (with that subject held out)
    geo = hyp.plot(trained_srms_dfs_run2[sub_idx][range(3)], title='SRM: subject %i held out, run 2.'% (sub_idx+1), 
                   group=digit_colors_run2, elev=elevations_per_plot[3], legend=run2_legend,
                   **figkwargs)


# In[ ]:


plot_all_for_subject_shared_responses(10)


# In[65]:


geo = hyp.plot(projected_dfs_run1[1] + projected_dfs_run2[1], size=[10,7], animate='parallel', show=False, frame_rate=15,save_path='animations_old/post_crossval_sr2.mp4' )
anim = geo.line_ani
# convert to html video and show
#HTML(anim.to_html5_video())


# In[ ]:




