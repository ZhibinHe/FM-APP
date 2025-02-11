###################################################
#  stage1:
#  using fmri to train feature extractor(fmri) to predict phenotypes
###################################################


import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from net.braingnn import Network_regress_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig
import pandas as pd
import pylab
from torch_geometric.data import Data, Dataset, DataLoader
from net.modeling_brainlm import *


config = BrainLMConfig()
torch.manual_seed(123)
EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/data/hzb/project/Brain_Predict_Score/data_hcp_4fmri/HCP_pcp/Gordon/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0005, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=333, help='feature dim')
parser.add_argument('--nroi', type=int, default=333, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--csvroot', type=str, default='/data/hzb/project/Brain_Predict_Score/ViTPre0219')
parser.add_argument('--datadir', type=str, default='/data/hzb/project/Brain_Predict_Score/ViTPre0219/VITPre_Pytorch_test_select/data')
parser.add_argument('--out_txt_name', type=str, default='output.txt')
parser.add_argument('--stage1_epochs', type=int, default=1000, help='number of epochs of training')  #52
parser.add_argument('--datatoot_t1w', type=str, default='/data/hzb/project/Brain_Predict_Score/BrainGNN/BrainGNN_Pytorch-main/data_hcp/HCP_pcp/Gordon/filt_noglobal', help='root directory of the dataset')


opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
path2 = opt.datatoot_t1w
name = 'HCP'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
stage1_epoch = opt.stage1_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))

class CustomDataset(Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, idx):
        a = torch.tensor(self.A[idx])
        b = torch.tensor(self.B[idx])
        return Data(x=a, y=b)




################## Define Dataloader ##################################
train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1 = data_load_hcp_t1fmri(path, path2, name, opt)  # load fmri t1w dataset
text_feature_all = text_feature
text_feature = text_feature[:, :28, :]
text_feature_zero = text_feature_all[1,28:,:].repeat(8, 1, 1)


train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)


model1 = CombinedModel_sink_feature().to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################

loss_fn = nn.MSELoss(reduction='none')


###################### Network Training Function#####################################
def train1(epoch):
    # print('train...........')
    scheduler1.step()

    model1.train()

    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer1.zero_grad()
        data.y = data.y[:,:28]
        score_predict , _, _ = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi+9)[:,:,:333], text_feature) #  fmri dataset, :333=fmri

        data.y = torch.tensor(data.y, dtype=torch.float)
        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T, data.y)
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c
        step = step + 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer1.step()



    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2






def test_acc1(loader):
    model1.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():

        pred_score = torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device)
        label_score = torch.tensor(torch.zeros(1, text_feature_all.shape[1]), device=device)
        fusion_feature = torch.tensor(torch.zeros(1, text_feature.shape[1], text_feature.shape[2] * 2),
                                      device=device_cpu)

        for data in loader:
            data = data.to(device)
            all_datay = data.y
            data.y = data.y[:, :28]

            score_predict, regress_weight, feature_cat = model1(
                data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi + 9)[:, :, :333], text_feature)  #

            score_predict1 = torch.stack(score_predict)[:, :, 0].T
            pred_score = torch.cat((pred_score, score_predict1), dim=0)
            label_score = torch.cat((label_score, all_datay), dim=0)
            fusion_feature = torch.cat((fusion_feature, feature_cat.cpu()), dim=0)

    for i in range(pred_score.shape[1]):
        correct_task = \
        np.corrcoef(pred_score[1:, i].detach().cpu().numpy().T, label_score[1:, i].detach().cpu().numpy().T)[0, 1]
        correct.append(correct_task)


    correct = np.sum(correct)/pred_score.shape[1]
    regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    regress_weight1 = regress_weight.cpu()
    del regress_weight

    return correct,  correct, fusion_feature[1:,:,:],  regress_weight1, label_score[1:,:], pred_score[1:,:] #/ len(loader.dataset)




def test_loss1(loader,epoch):
    # print('testing...........')
    model1.eval()
    loss_all = 0
    loss_c = []
    for data in loader:
        data = data.to(device)
        data.y = data.y[:, :28]
        score_predict , _ , _ = model1(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi+9)[:,:,:333], text_feature)   #
        data.y = torch.tensor(data.y, dtype=torch.float)
        column_losses = loss_fn(torch.stack(score_predict)[:,:,0].T, data.y)
        loss_c = torch.sum(column_losses)
        loss = opt.lamb0*loss_c #+ opt.lamb1 * correlation_loss #+ opt.lamb2 * loss_p2 \
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

############################   Model Training #########################################
best_model_wts = copy.deepcopy(model1.state_dict())
best_loss = 1e10
val_acc_list = []


for epoch in range(0,stage1_epoch):
    since  = time.time()
    tr_loss= train1(epoch)


    tr_acc, tr_rmse, fusion_feature_tr,  regress_weight_tr, label_score_tr, pred_score_tr= test_acc1(train_loader)
    val_acc, val_rmse, fusion_feature_val,  regress_weight_val, label_score_val, pred_score_val = test_acc1(val_loader)
    val_acc_list.append(val_acc)
    val_loss = test_loss1(val_loader,epoch)
    time_elapsed = time.time() - since
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f},Train rmse: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test rmse: {:.7f}, '.format(epoch, tr_loss,
                                                       tr_acc, tr_rmse, val_loss, val_acc, val_rmse))





# torch.save(model1, './model/stage1_fmri_best_0602.pth')
# torch.save([train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val], './model/stage1_dataset_0602.pt')





