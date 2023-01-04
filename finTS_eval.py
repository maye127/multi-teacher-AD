
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import MY_Data
import argparse

from finTS import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'cifar10'])
parser.add_argument('--filepath', type=str, default='./run')
# parser.add_argument('--student', type=str, default='resnet', choices=['resnet', 'vgg', 'densenet'])
opt = parser.parse_args()





# compute results on different anomaly score, find max average AUC.
res = list()
for normal_lab in range(15 if opt.dataset == 'mvtec' else 10):
    if opt.dataset == 'cifar10':
        myCIFAR10 = MY_Data.CIFAR10()
        [_, data_loader_test, label_test] = myCIFAR10.get(normal_lab=normal_lab, batch_size=1)
        AS_z_res, AS_cossim_res, AS_rec_res = torch.load(opt.filepath + '/cifar_res/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
        AS_z_vgg, AS_cossim_vgg, AS_rec_vgg = torch.load(opt.filepath + '/cifar_vgg/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
        AS_z_dense, AS_cossim_dense, AS_rec_dense = torch.load(opt.filepath + '/cifar_dense/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
    if opt.dataset == 'mvtec':
        myMVTec = MY_Data.MVTec()
        [_, data_loader_test, label_test] = myMVTec.get_1024(normal_lab=normal_lab, batch_size=1)
        AS_z_res, AS_cossim_res, AS_rec_res = torch.load(opt.filepath + '/mvtec_res/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
        AS_z_vgg, AS_cossim_vgg, AS_rec_vgg = torch.load(opt.filepath + '/mvtec_vgg/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
        AS_z_dense, AS_cossim_dense, AS_rec_dense = torch.load(opt.filepath + '/mvtec_dense/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
    # AS_z, AS_cossim, AS_rec = torch.load(opt.filepath + '/anomaly_score' + str(normal_lab) + '.data', map_location='cpu')
    AS_z = normalization(AS_z_res) + normalization(AS_z_vgg) + normalization(AS_z_dense)
    AS_cossim = normalization(AS_cossim_res) + normalization(AS_cossim_vgg) + normalization(AS_cossim_dense)
    AS_rec = normalization(AS_rec_res) + normalization(AS_rec_vgg) + normalization(AS_rec_dense)
    res_class = list()
    res_class.append(normal_lab)
    list_as = [AS_z, AS_cossim, AS_rec]
    columns = ['normal_lab', 'AS_z', 'AS_cossim', 'AS_rec']
    for r1 in torch.linspace(0, 1, 11):
        for r2 in torch.linspace(0, (1 - r1), int((1 - r1) / 0.1 + 1)):
            r3 = 1 - r1 - r2
            list_as.append(r1 * AS_z + r2 * AS_cossim + r3 * AS_rec)
            # 改进 list_as.append(r1 * normalization(AS_z) + r2 * normalization(AS_cossim) + r3 * normalization(AS_rec))
            columns.append(str(r1) + str(r2) + str(r3))
    for Anomaly_Score in list_as:
        Anomaly_Score = Anomaly_Score.reshape(label_test.size, -1)
        Anomaly_Score = torch.max(Anomaly_Score, dim=1).values
        AUC = roc_auc_score(y_true=label_test, y_score=Anomaly_Score)
        res_class.append(AUC)
    res.append(np.stack(res_class))
res = np.stack(res)

res_mean = res.mean(axis=0)
res_mean[0] = 0
res_mean.max()
print(res_mean)
print(res_mean.argmax(), columns[res_mean.argmax()], res_mean.max())
res = np.concatenate([res, res_mean[np.newaxis, :]], axis=0)
pd.DataFrame(columns=columns, data=res).to_csv(opt.filepath + '/eval_' + opt.dataset + '.csv', encoding='utf-8')
