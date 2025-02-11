from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
import numpy as np



# subject_id = np.loadtxt('/data/hzb/project/Brain_Predict_Score/data_hcp_4fmri/HCP_pcp/Gordon/Subject_ID.txt', delimiter=',')
subject_id = np.genfromtxt('/data/hzb/project/Brain_Predict_Score/HCPA/data_preprocess_fmri_4/HCPA_pcp/Gordon/Subject_ID.txt', delimiter=',', dtype = str)

for i in range(len(subject_id)):
    folder_name = str(subject_id[i][: -1])
    os.makedirs('/data/hzb/project/Brain_Predict_Score/HCPA/data_preprocess_fmri_4/HCPA_pcp/Gordon/filt_noglobal/'+folder_name)  # makedirs 创建文件时如果路径不存在会创建这个路径





