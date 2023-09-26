#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Age classification fitting both the age groups and generate the training
and testing data matrix for LOO classification"""

# In[3]:


import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import time
import csv
import numpy as np
from brainiak.funcalign.rsrm import RSRM
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import math
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

get_ipython().run_line_magic('autosave', '5')

# In[4]:

"""importing young subject names"""

with open('/Users/akalyani/Desktop/projects/somato_align/scripts/young_subjects.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    young = list(zip(*reader))[0]

"""importing old subject names"""
with open('/Users/akalyani/Desktop/projects/somato_align/scripts/old_subjects.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    old = list(zip(*reader))[0]

age = [young, old]

loo = LeaveOneOut()

# In[5]:


ds_dir = '/Volumes/RG_Kuehn/Avinash/cSRM/sense_data'
# In[6]:
# labels for training the classifier, 0 for young and 1 for old
l1 = list(np.repeat(0, (19)))
l2 = list(np.repeat(1, (19)))
labels = np.concatenate((l1, l2))
labels = list(labels)


def run_crossval_both_proj(arrs1_y, arrs2_y,
                           arrs1_o, arrs2_o,
                           k, vox,
                           niter=30, ):
    # prepare empty results array # shape (nlabels, nruns, nsubs)
    nruns = 2  # number of runs
    nsubs = len(arrs1_y)

    """easy to call arrays"""
    arrs1_yo = [arrs1_y, arrs1_o]
    arrs2_yo = [arrs2_y, arrs2_o]
    # proj_age_run_dat =[]
    for age_idx in range(2):
        run1_arrs = arrs1_yo[age_idx]
        run2_arrs = arrs2_yo[age_idx]

        proj_run_dat_1 = []

        subjs = int(len(run1_arrs))

        for trainrun_idx in range(2):
            projected_data_1 = list(np.zeros(nsubs))
            for testsub_idx in range(subjs):  # iterate over runs
                # select run used for training and test and according digit indices
                training_arrs_1 = (run1_arrs, run2_arrs)[trainrun_idx]
                training_arrs_2 = (arrs1_yo[abs(age_idx - 1)], arrs2_yo[abs(age_idx - 1)])[trainrun_idx]
                test_arrs_1 = (run1_arrs, run2_arrs)[abs(trainrun_idx - 1)]
                # we need only one test_data
                # test_arrs_2 = ((arrs1_yo[abs(age_idx - 1)], arrs2_yo[abs(age_idx - 1)]))[abs(trainrun_idx - 1)]

                start = time.time()
                print('starting run %i subject %i and age %i' % (trainrun_idx, testsub_idx, age_idx))

                # testsub_idxs = [testsub_idx_c1, testsub_idx_c2]
                trainsubs_traindata_1 = [x for i, x in enumerate(training_arrs_1) if
                                         i != testsub_idx]  # select training data
                trainsubs_traindata_2 = [x for i, x in enumerate(training_arrs_2) if i != testsub_idx]
                testsubs_traindata_1 = training_arrs_1[testsub_idx]
                # transform for 1st group:
                srm_1 = RSRM(n_iter=niter, features=k)  # train srm on training subject's training data
                srm_1.fit((trainsubs_traindata_1))
                w_1, s_1 = srm_1.transform_subject((testsubs_traindata_1))  # estimate test subject's bases
                srm_1.w_.insert(testsub_idx, w_1)
                srm_1.s_.insert(testsub_idx, s_1)
                # transform for 2nd group:
                srm_2 = RSRM(n_iter=niter, features=k)  # train srm on training subject's training data
                srm_2.fit((trainsubs_traindata_2))
                w_2, s_2 = srm_2.transform_subject((testsubs_traindata_1))  # estimate test subject's bases
                srm_2.w_.insert(testsub_idx, w_2)
                srm_2.s_.insert(testsub_idx, s_2)

                # transforming the testsubject's test_data #whole testarray
                shared_test_1, ind_terms_1 = srm_1.transform(test_arrs_1)  # project test run into shared space
                shared_test_2, ind_terms_2 = srm_2.transform(test_arrs_1)  # project test run into shared space
                projected_data_1[testsub_idx] = np.concatenate((shared_test_1[testsub_idx], shared_test_2[testsub_idx]),
                                                               axis=1)
                elapsed = time.time() - start
                print('this round took: ', elapsed)
                print(len(projected_data_1), projected_data_1[testsub_idx].shape)
            proj_run_dat_1.append(projected_data_1)
        print(len(proj_run_dat_1))
        print(vox, age_idx, k)
        np.savez('/Volumes/IKND/AG_Kuehn/Avinash/new_final_results/AGE_projection_rSRM_permuted_2_%s_%s_%s.npz' % (
        vox, age_idx, k), projected_data=proj_run_dat_1)

    return None


# In[51]:

def loo_classify_balanced_comb(g1, g2, labels):
    loo = LeaveOneOut()
    X1, X2 = [], []
    for i in range(len(g1)):
        x1 = g1[i].flatten()
        x2 = g2[i].flatten()
        X1.append(x1)
        X2.append(x2)
    print(len(X1), X1[0].shape)
    print(len(X2), X2[0].shape)
    all_r1 = np.concatenate((X1, X2), axis=0)
    # all_r1 = np.nan_to_num(stats.zscore(all_r, axis=0, ddof=1))
    all_score2 = []
    for train_idx, test_idx in loo.split(X1):
        X1_train, X1_test = np.array(X1)[train_idx.astype(int)], np.array(X1)[test_idx.astype(int)]
        y1_train, y1_test = np.array(l1)[train_idx.astype(int)], np.array(l1)[test_idx.astype(int)]
        all_score = []
        for train_idx2, test_idx2 in loo.split(X2):
            X2_train, X2_test = np.array(X2)[train_idx2.astype(int)], np.array(X2)[test_idx2.astype(int)]
            X_test = np.concatenate((X1_test, X2_test))
            X_train = np.concatenate((X1_train, X2_train))

            y2_train, y2_test = np.array(l2)[train_idx2.astype(int)], np.array(l2)[test_idx2.astype(int)]
            y_train = np.concatenate((y1_train, y2_train))
            y_test = np.concatenate((y1_test, y2_test))
            # classifier = NuSVC(nu=0.5, kernel='rbf', gamma='auto')
            classifier = svm.SVC(kernel='linear')
            classifier = classifier.fit(X_train, y_train)
            predicted_labels = classifier.predict(X_test)
            score = accuracy_score(y_test, predicted_labels)
            all_score.append(score)

        # print('class',predicted_labels, 'true', y_test, score)
        all_score2.append(all_score)
    print(np.mean(all_score2))
    mean_score = np.mean(all_score2)
    stdv = np.std(all_score2)
    cv = StratifiedKFold(2, shuffle=True)
    cv_scr, per_score, pval = permutation_test_score(classifier, all_r1, labels, scoring="accuracy", cv=cv,
                                                     n_permutations=1000)
    return mean_score, stdv, cv_scr, per_score, pval


# In[27]:


with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/run1_AGE_young_old_hemi.npz', allow_pickle=True) as data:
    arrs1_all = data['run1_arrs']
    arrs2_all = data['run2_arrs']
arrs1_all[3, 1][0] = np.pad(arrs1_all[3, 1][0], ((0, 1), (0, 0)), 'constant')
arrs1_all[4, 1][0] = np.pad(arrs1_all[4, 1][0], ((0, 1), (0, 0)), 'constant')
arrs2_all[3, 1][0] = np.pad(arrs1_all[3, 1][0], ((0, 1), (0, 0)), 'constant')
arrs2_all[4, 1][0] = np.pad(arrs1_all[4, 1][0], ((0, 1), (0, 0)), 'constant')

# In[52]:


the_voxels = [501, 1001, 1501, 2001, 2501, 3001]

# In[56]:


# WB age classfication
l1 = list(np.repeat(0, (19)))
l2 = list(np.repeat(1, (19)))
labels_age = np.concatenate((l1, l2))
labels_group = list(labels_age)

# In[53]:


# age, sub, label CONSERVATIVE
k = 20
cv_scores = list(np.zeros(shape=(len(the_voxels), 1)))
accu_labels = list(np.zeros(len(the_voxels)))
for vox_idx, vox in enumerate(the_voxels):
    v = vox - 1
    young_r1 = [arrs1_all[vox_idx][0][i][:v, :] for i in range(19)]
    # print(young_r1[0].shape)
    old_r1 = [arrs1_all[vox_idx][1][i][:v, :] for i in range(19)]
    # print(old_r1[0].shape)
    young_r2 = [arrs2_all[vox_idx][0][i][:v, :] for i in range(19)]
    # print(young_r2[0].shape)
    old_r2 = [arrs2_all[vox_idx][1][i][:v, :] for i in range(19)]
    # print(old_r2[0].shape)
    run_crossval_both_proj(young_r1, young_r2,
                           old_r1, old_r2,
                           k, vox,
                           niter=30, )

# In[54]:


# load the accuracies for plotting
for vox_idx, vox in enumerate(the_voxels):
    with np.load('/Volumes/IKND/AG_Kuehn/Avinash/new_final_results/AGE_projection_rSRM_permuted_2_%s_%s_%s.npz' % (
    vox, 0, 20),
                 allow_pickle=True) as data:
        projected_data = data['projected_data']
    shared_all_y1 = projected_data[0]
    shared_all_y2 = projected_data[1]

    with np.load('/Volumes/IKND/AG_Kuehn/Avinash/new_final_results/AGE_projection_rSRM_permuted_2_%s_%s_%s.npz' % (
    vox, 1, 20),
                 allow_pickle=True) as data:
        projected_data = data['projected_data']
    shared_all_o1 = projected_data[0]
    shared_all_o2 = projected_data[1]
    a, b, c, d, e = loo_classify_balanced_comb(shared_all_y1, shared_all_o1, labels)
    cv_scores[vox_idx] = [a, b, c, d, e]

# In[83]:


np.savez('/Volumes/IKND/AG_Kuehn/Avinash/new_final_results/AGE_SRM_hemi_classification_scores_permuted.npz',
         accu_scores=cv_scores)


# In[ ]:


# In[84]:


# calculating standard error for raw 
def calculate_standard_error(mean, standard_deviation, sample_size):
    standard_error = standard_deviation / math.sqrt(sample_size)
    return standard_error


# In[33]:


SE_raw = [calculate_standard_error(raw_accu[i], raw_std[i], 38) for i in range(len(raw_accu))]

# In[ ]:


# In[85]:


with np.load('/Volumes/IKND/AG_Kuehn/Avinash/new_final_results/AGE_SRM_hemi_classification_scores_permuted.npz',
             allow_pickle=True) as data:
    accu = data['accu_scores']

# In[86]:


# mean_score, stdv , cv_scr, per_score, pval

cv_accu = [accu[i][0] for i in range(len(the_voxels))]
cv_std = [accu[i][1] for i in range(len(the_voxels))]

SE_cv = [calculate_standard_error(cv_accu[i], cv_std[i], 38) for i in range(len(cv_accu))]

# In[87]:


# load the accuracies for plotting
with np.load('/Volumes/AvinashPhD/cSRM/FINAL_RESULTS/AGE_SRM_hemi_classification.npz',
             allow_pickle=True) as data:
    raw_accu = data['raw_accu']
    raw_std = data['raw_std']

# In[89]:


SE_raw = [calculate_standard_error(raw_accu[i], raw_std[i], 38) for i in range(len(raw_accu))]

# In[68]:


# plot data in grouped manner of bar type
vox_strings = ['500', '1000', '1500', '2000', '2500', '3000']

# In[111]:


N = 6
a1_means = np.asarray(raw_accu)
a1_std = np.asarray(SE_raw)

ind = np.arange(N)  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, a1_means, width, color='b', yerr=a1_std)

a2_means = cv_accu
a2_std = SE_cv
rects2 = ax.bar(ind + width, a2_means, width, color='c', yerr=a2_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores', size=20)
ax.set_title('Scores by voxels and methods', size=18)
ax.set_xticks(ind + width / 2)
ax.set_xlabel('Voxels', size=20)
ax.set_xticklabels(vox_strings)

ax.legend((rects1[0], rects2[0]), ('Raw Voxel', 'rSRM'), fontsize=14, loc=3)


plt.xticks(size=17)
plt.yticks(size=17)
plt.axhline(y=0.5, color='r', linestyle='--')

fig.savefig('/Volumes/IKND/AG_Kuehn/Avinash/FINAL_PLOTS/RAW_vs_AGE_perm2.png', dpi=300, bbox_inches='tight')
plt.show()