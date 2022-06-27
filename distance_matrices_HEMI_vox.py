#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from os.path import join as pjoin
import numpy as np
import pandas as pd
import runpy
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# import functions from other scripts
file_globals = runpy.run_path('digit_classification_knn.py')
get_digit_indices = file_globals['get_digit_indices']


# In[2]:


#import projections
feats= [5]


# In[3]:


def distance_map(proj_data_run1train, trained_srms_run1train, proj_data_run2train, trained_srms_run2train):
    # create finger-color arrays

    # get ndarrays (nfingers, nvols)
    digits_run1, digits_run2 = get_digit_indices()
    #digits_run2=digits_run1###############################
    # turn into flat arrays with values 1-5
    digit_colors_run1, digit_colors_run2 = np.zeros(shape=(256)), np.zeros(shape=(256))
    for finger_i in range(1,6):
        digit_colors_run1[digits_run1[finger_i-1]] = finger_i
        digit_colors_run2[digits_run2[finger_i-1]] = finger_i

    # for some reason, the first entry in the first run's digit array is a 5, but should be a 1...
    digit_colors_run1[0]=1.
    # make lists of data frames

    # first run is training, second is test
    projected_dfs_run2 = []
    trained_srms_dfs_run1 = []

    for sub_idx in range(proj_data_run1train.shape[0]):
        df = pd.DataFrame(proj_data_run1train[sub_idx].T)
        projected_dfs_run2.append(df)
        #srm_df = pd.DataFrame(trained_srms_run1train[sub_idx].T)
        #trained_srms_dfs_run1.append(srm_df)

    # second run is training, first is test
    projected_dfs_run1 = []
    trained_srms_dfs_run2 = []

    for sub_idx in range(proj_data_run2train.shape[0]):
        df = pd.DataFrame(proj_data_run2train[sub_idx].T)
        projected_dfs_run1.append(df)
        #srm_df = pd.DataFrame(trained_srms_run2train[sub_idx].T)
        #trained_srms_dfs_run2.append(srm_df)
    """
    Gather matrices for all subjects and save in big array
    """
    # pick out number of dimensions and volumes
    ndims, nvols = proj_data_run1train[0].shape

    # set up results array
    timepoint_distances = np.zeros(shape=(2, len(proj_data_run1train), nvols, nvols))

    # iterate over runs and subjects
    for run_idx, dataset in enumerate([proj_data_run1train, proj_data_run2train]):
        for sub_idx in range(len(proj_data_run1train)):

            for vol_i in range(nvols):
                for vol_j in range(nvols):
                    dist = euclidean(dataset[sub_idx][:,vol_i],
                                     dataset[sub_idx][:,vol_j])
                    timepoint_distances[run_idx, sub_idx, vol_i, vol_j] = dist
    """
    Now average along subject and run axes
    """
    av_timepoint_dist = np.average(timepoint_distances, axis=(0,1))
    test_df = projected_dfs_run1[0]
    # add correct digit labels for given run
    test_df['digit'] = digit_colors_run1#[:253]
    # prepare results arrays
    av_distances = np.zeros(shape=(2, len(projected_dfs_run1), 5 , 5))
    std_distances = np.zeros(shape=(2, len(projected_dfs_run1), 5 , 5))

    # iterate over runs and subjects
    for run_idx in range(2):
        # select correct run and corresponding finger labels
        projected_dfs = [projected_dfs_run1, projected_dfs_run2][run_idx]
        #digit_labels = [digit_colors_run1[:253], digit_colors_run2[:253]][run_idx]
        digit_labels = [digit_colors_run1, digit_colors_run2][run_idx]
        for sub_idx in range(len(projected_dfs_run1)):
            # select correct data frame and add digit labels for slicing
            df = projected_dfs[sub_idx]
            df['digit'] = digit_labels
            # iterate over pairs of digits
            for source_digit in range(1,6):
                for target_digit in range(1,6):
                    # select subset of data frame for source and target digit (array of shape nsamples, nfeatures)
                    source_data = df[df.digit == source_digit].drop(columns=['digit']).values
                    target_data = df[df.digit == target_digit].drop(columns=['digit']).values
                    # iterate over pairs of samples, compute distance, and save in list
                    sample_wise_distances = []
                    for source_sample in range(source_data.shape[0]):
                        for target_sample in range(target_data.shape[0]):
                            dist = euclidean(source_data[source_sample], target_data[target_sample])
                            sample_wise_distances.append(dist)
                    # compute average and standard deviation of these distances
                    av_distances[run_idx, sub_idx, source_digit-1, target_digit-1] = np.average(sample_wise_distances)
                    std_distances[run_idx, sub_idx, source_digit-1, target_digit-1] = np.std(sample_wise_distances)
                                
                
    return av_timepoint_dist, av_distances, std_distances
    


# In[4]:


"""YOUNG PROJECTION"""
the_voxels = [501, 1001, 1501, 2001,2501,3001]
euc_dist_vox, agg_vox =[], []
for vox in the_voxels:
    euc_dist_age, agg_age = [], []
    for age_idx in range(2):
        euc_dist_feat, agg_feat = [], []
        for n_feat in feats:
            with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/projection_%s_%s_%s.npz' %(vox,age_idx,n_feat), allow_pickle = True) as data:
                projected_data = data['proj_data']
                trained_srms = data['srms']
                
            trained_srms_run1train = trained_srms[0]
            proj_data_run1train = projected_data[0]
            proj_data_run2train = projected_data[1]
            trained_srms_run2train = trained_srms[1]
            euc_dist, av_distances, std_distances=distance_map(proj_data_run1train, trained_srms_run1train, proj_data_run2train, trained_srms_run2train)
            agg_av_distances = np.average(av_distances, axis=(0,1))
            agg_std_distances = np.average(std_distances, axis=(0,1))
            agg_feat.append(agg_av_distances)
            euc_dist_feat.append(euc_dist)
        euc_dist_age.append(euc_dist_feat)
        agg_age.append(agg_feat)
    euc_dist_vox.append(euc_dist_age)
    agg_vox.append(agg_age)
            


# In[5]:


"""DISTANCE MEASURES FOR BAR/VIOLIN PLOT"""
the_voxels = [501]
euc_dist_vox, agg_vox =[], []
for vox in the_voxels:
    euc_dist_age, agg_age = [], []
    for age_idx in range(2):
        euc_dist_feat, agg_feat = [], []
        for n_feat in feats:
            with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/projection_%s_%s_%s.npz' %(vox,age_idx,n_feat), allow_pickle = True) as data:
                projected_data = data['proj_data']
                trained_srms = data['srms']
                
            trained_srms_run1train = trained_srms[0]
            proj_data_run1train = projected_data[0]
            proj_data_run2train = projected_data[1]
            trained_srms_run2train = trained_srms[1]
            euc_dist, av_distances, std_distances=distance_map(proj_data_run1train, trained_srms_run1train, proj_data_run2train, trained_srms_run2train)
            agg_av_distances = np.average(av_distances, axis=(0,1))
            agg_std_distances = np.average(std_distances, axis=(0,1))
            agg_feat.append(agg_av_distances)
            euc_dist_feat.append(av_distances)
        euc_dist_age.append(euc_dist_feat)
        agg_age.append(agg_feat)
    euc_dist_vox.append(euc_dist_age)
    agg_vox.append(agg_age)


# In[6]:


agg_vox[0][1][0]


# In[7]:


# aggregate (average) over subjects and runs
agg_av_distances = np.average(euc_dist_vox[0][0][0], axis=(0,1))
agg_std_distances = np.std(euc_dist_vox[0][0][0], axis=(0,1))
matrix = agg_av_distances.T
matrix2 = agg_std_distances.T
diags_y = [matrix.diagonal(i)for i in range(5)]
diags2_y = [matrix2.diagonal(i)for i in range(5)]
NN = np.zeros(5)
NN2 = np.zeros(5)
for i in range(5):
    NN[i] = np.mean(diags_y[i])
    NN2[i] = np.mean(diags2_y[i])
NN_young = NN
NN2_young = NN2


# In[8]:


# aggregate (average) over subjects and runs
agg_av_distances = np.average(euc_dist_vox[0][1][0], axis=(0,1))
agg_std_distances = np.std(euc_dist_vox[0][1][0], axis=(0,1))
matrix = agg_av_distances.T
matrix2 = agg_std_distances.T
diags_o = [matrix.diagonal(i)for i in range(5)]
diags2_o = [matrix2.diagonal(i)for i in range(5)]
NN = np.zeros(5)
NN2 = np.zeros(5)
for i in range(5):
    NN[i] = np.mean(diags_o[i])
    NN2[i] = np.mean(diags2_o[i])
NN_old = NN
NN2_old = NN2


# In[9]:


NN_young


# In[10]:


x = np.arange(5)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(5,4))
rects1 = ax.bar(x - width/2, np.asarray(NN_young).T, width, yerr=np.asarray(NN2_young).T,label='young', color = 'blue')
rects2 = ax.bar(x + width/2, np.asarray(NN_old).T, width,yerr=np.asarray(NN2_old).T, label='old',color = 'red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Eucledian distance',fontsize=15)
ax.set_title('Euclidean distance averaged per neighbour',fontsize=15)
ax.set_xlabel('Neighbours',fontsize=15)
ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/NN_dis.png', bbox_inches='tight', dpi=300)
plt.style.use('default')
plt.show()


# In[ ]:





# In[73]:


#extracting the diagonals for all the subjects
diag_y_all , diag_o_all = [],[]
for i in range(19):
    diags = euc_dist_vox[0][0][0][0][i].diagonal(0)
    diags2 = euc_dist_vox[0][1][0][0][i].diagonal(0)
    
    diag_y_all.append(diags)
    diag_o_all.append(diags2)


# In[74]:


D_y_all, D_o_all = [],[]
for d in range(5):
    D_y = [diag_y_all[i][d].T for i in range(19)]
    D_o = [diag_o_all[i][d].T for i in range(19)]
    
    D_y_all.append(D_y)
    D_o_all.append(D_o)
    


# In[79]:


df_dists_y = pd.DataFrame(np.array(D_y_all).T, columns = ["D1", 'D2', 'D3', 'D4', 'D5'])
df_dists_o = pd.DataFrame(np.array(D_o_all).T, columns = ["D1", 'D2', 'D3', 'D4', 'D5'])


# In[80]:


df_dists_new_y = df_dists_y.melt(var_name='Column', value_name='Value')
df_dists_new_o = df_dists_o.melt(var_name='Column', value_name='Value')


# In[84]:


df_all = df_dists_new_y.append(df_dists_new_o)
df_all['age'] = [*['young']*95,*['old']*95]


# In[85]:


df_all


# In[89]:


sns.violinplot(x="Column", y="Value", hue="age",data=df_all, palette="muted")
#sns.swarmplot(x="Column", y="Value", hue="age",data=df_all, palette="muted")


# In[87]:


sns.barplot(x="Column", y="Value", hue="age",data=df_all, palette="muted")


# In[82]:


ax = sns.violinplot(x = "Column", y = "Value", data = df_dists_new_y)


# In[78]:


ax = sns.barplot(x = "Column", y = "Value", data = df_dists_new)


# In[13]:



matrix=agg_vox[0][1][0]
diags = [matrix.diagonal(i)for i in range(5)]


# In[14]:


diags


# In[18]:


# mask lower triangle (because matrices are symmetric)
tril_mask = np.zeros_like(matrix, dtype=np.bool)
tril_mask[np.triu_indices_from(tril_mask, k=1)] = True

# start plotting
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12,5))
sns.set(font_scale=1.3)
digit_strings = ['D-%i' %i for i in range(1,6)]
shared_figkws1 = {'cmap':'cividis', 'fmt':'.1f','cbar':False,'vmin':8,'vmax':16,'annot':True,'square':False,
                 'xticklabels':digit_strings, 'yticklabels':digit_strings, 'mask':tril_mask}
shared_figkws2 = {'cmap':'cividis', 'fmt':'.1f','cbar':True,'vmin':8,'vmax':16,'annot':True, 'square':False,
                 'xticklabels':digit_strings, 'yticklabels':digit_strings, 'mask':tril_mask}

sns.heatmap(agg_vox[0][0][0], ax=axs[0], **shared_figkws1)
sns.heatmap(agg_vox[0][1][0], ax=axs[1], **shared_figkws2)

supttl = plt.suptitle("""Digit-wise average euclidean distance between samples in shared space""", fontsize = 18
            )
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
#l, r = plt.xlim()
#l += 0.5 # Add 0.5 to the bottom
#r += 0.5 # Subtract 0.5 from the top

axs[0].set_title('Young', fontsize=16)

axs[1].set_title('Old',fontsize=16)
#axs[0].set_ylim(b, t)
#axs[1].set_ylim(b, t)
#axs[1].set_xlim(l, r)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(labelsize=14)
#cb.axs[1].
#fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/Dis_2.png', bbox_inches='tight',bbox_extra_artists=[supttl], dpi=300)
plt.show() # ta-da!


# In[69]:


fig = plt.figure(figsize=(8,7))
axis = sns.heatmap(euc_dist_vox[0][0][0], cbar=True, cmap='cividis')
plt.title("""Euclidean distance between time points in
shared space averaged over subjects and runs""")
axis.set_xticks([])
axis.set_yticks([])
#fig.savefig('/Volumes/AvinashPhD/cSRM/FINAL_PLOTS/Dis_y_1.png',dpi=300)
plt.show()


# In[ ]:




