#!/bin/sh

for file in *;

do echo $file;
antsApplyTransforms --interpolation NearestNeighbor -d 3 -e 3 -i $file/D1_D5/adata1.nii -r $file/mri/brain.nii -t $file/mri/init_auto.txt -o $file/coreg1.nii.gz
antsApplyTransforms --interpolation NearestNeighbor -d 3 -e 3 -i $file/D5_D1/adata2.nii -r $file/mri/brain.nii -t $file/mri/init_auto.txt -o $file/coreg2.nii.gz
done
