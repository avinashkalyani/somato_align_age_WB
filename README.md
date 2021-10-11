# somato_align_age_WB
SRM of the S1 with age differences and in the WB as well 
Steps:-(Preparing the datasets for SRM)

Step1: Preprocessing with SPM_batch_analyses_combined.m will be used as a localizer 
- (Input: subj/blocked_design/sess1_and_sess2; Output: subj/blocked_design/Analyses/ )/n

Step2: Parcellation of T1.nii (recon-all -s sub1 -i /Volumes/TRANSFER/Avinash_somato/all_data/bcs/T1.nii -all)
- (Input: path_to_subj/T1.nii Output: sub1/label)
Step3: Coregistration 
