# somato_align_age_WB
SRM of the S1 with age differences and in the WB as well 

Steps:-(Preparing the datasets for SRM)

Step1: Preprocessing with SPM_batch_analyses_combined.m will be used as a localizer 
- (Input: subj/blocked_design/sess1_and_sess2; Output: subj/blocked_design/Analyses/ )

Step2.a: Parcellation of T1.nii (recon-all -s sub1 -i /Volumes/TRANSFER/Avinash_somato/all_data/bcs/T1.nii -all)
- (Input: path_to_subj/T1.nii Output: sub1/label)

Step2.b: Label to volume conversion

Step3: Coregistration ants_transform_all.sh
- (Input:adata.nii(functional data), subj/mri/brain.nii(skull-stripped anatomical data generated after recon-all), subj/mri/init_auto.txt(itk-snap manual registration matrix)) 
- (Output: subj/coreg1.ni.gz,coreg2.nii.gz for run1 and 2 respectively)

After obtaining the dataset arrays post masking, 
- Age group prediction : age_classification.py
- Digit-wise euclidean distance plots:  distance_matrices_HEMI_vox.py

