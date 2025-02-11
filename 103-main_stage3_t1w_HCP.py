###################################################
#  stage3:
#  regressor synthesizer and zero-shot inference
###################################################




import os
import numpy as np
import argparse
import time
import copy
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.data_load_hcp import *
from net.configuration_brainlm import BrainLMConfig

import pandas as pd
import pylab

# 333 node Epoch: 018 0.1822354,  0m 7s
from torch_geometric.data import Data, Dataset, DataLoader
from net.modeling_brainlm import *

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
parser.add_argument('--csv_head_num', type=int, default=3)
parser.add_argument('--out_txt_name', type=str, default='output.txt')
parser.add_argument('--stage1_epochs', type=int, default=9, help='number of epochs of training')  #52
parser.add_argument('--datatoot_t1w', type=str, default='/data/hzb/project/Brain_Predict_Score/BrainGNN/BrainGNN_Pytorch-main/data_hcp/HCP_pcp/Gordon/filt_noglobal', help='root directory of the dataset')

# bs=8, lamb0=1,  epoch=8, 0.2043

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

################## Define Dataloader ##################################

train_loader, val_loader,text_feature, text_feature_zero, fusion_feature_tr, fusion_feature_val, label_score_tr, label_score_val, regress_weight_tr, regress_weight_val = torch.load('./model/stage2_dataset_0602.pt')
train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1 = data_load_hcp_t1fmri(path, path2, name, opt)
text_feature_all = text_feature
text_feature = text_feature[:, :28, :]
text_feature_zero = text_feature_all[1,28:,:].repeat(8, 1, 1)



model2 =BrainLMDecoder_mask(config, num_patches=196).to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)

scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################

loss_fn = nn.MSELoss(reduction='none')
loss_rec = nn.L1Loss(reduction='none')



def train2(epoch, train_loader_stage2):
    scheduler2.step()

    model2.train()

    loss_all = 0
    step = 0
    for data in train_loader_stage2:
        data = data.to(device)
        optimizer2.zero_grad()

        fusion_feature_sbj = data.x.view(opt.batchSize, 28, 1024)
        label_sbj = data.y.view( opt.batchSize, 51)       #28
        merged_matrix = torch.cat([fusion_feature_sbj, regress_weight_tr_repeat], dim=1)
        odd_columns = torch.arange(1, merged_matrix.size(1), 2)
        merged_matrix[:, odd_columns, :] = fusion_feature_sbj
        even_columns = torch.arange(0, merged_matrix.size(1), 2)
        merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
        # merged_matrix = merged_matrix.view(merged_matrix.shape[0], 1, merged_matrix.shape[1], merged_matrix.shape[2])

        # merged_matrix = torch.cat([regress_weight_tr_repeat, regress_weight_tr_repeat], dim=1)
        # xyz_vectors = merged_matrix[:,:,0:3]
        # noise = torch.rand(opt.batchSize, 56, device=device)

        # out = model2(merged_matrix, xyz_vectors, noise)


        out, mask = model2(merged_matrix, model2.training, 0)
        # loss11 = abs((out.logits[:, :, 0, :]-merged_matrix)).sum() *0.01
        # np.corrcoef(out.logits[0, 0, 0, :].detach().cpu().numpy(), merged_matrix[0, 0, :].detach().cpu().numpy())
        t1 = mask.view(8,56,1)
        t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
        # torch.matmul(merged_matrix[0, 17, :].T, merged_matrix[0, 16, :])    torch.matmul(merged_matrix[0, 17, :].T, out.logits[0, 16, 0, :])
        loss12 = abs((out.logits - merged_matrix.view(8,56,1,1024)) * t1).sum()
        loss11 = abs((out.logits[:, :, 0, :]-merged_matrix)).sum() *0.01


        #, xyz_vectors, noise
        # loss11 = out.loss
        # print(loss11)

        loss = opt.lamb0*loss11 +loss12
        step = step + 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=4.0)
        loss_all += loss.item() * data.num_graphs
        optimizer2.step()


    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=4.0)


def test_acc2(train_loader_stage2):
    model2.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():
      label_train1_corr  =  torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)
      label_sys_corr =  torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)
      train1_sys_corr =   torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)

      label_score  =  torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)
      train1_score =  torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)
      sys_score =   torch.tensor(torch.zeros(1, text_feature.shape[1]), device=device_cpu)

      for data in train_loader_stage2:
          data = data.to(device)
          optimizer2.zero_grad()

          fusion_feature_sbj = data.x.view(opt.batchSize, 28, 1024)
          label_sbj = data.y.view(opt.batchSize, 51)  #28
          label_sbj = label_sbj[:,:28]
          merged_matrix = torch.cat([fusion_feature_sbj, regress_weight_tr_repeat], dim=1)
          odd_columns = torch.arange(1, merged_matrix.size(1), 2)
          merged_matrix[:, odd_columns, :] = fusion_feature_sbj
          even_columns = torch.arange(0, merged_matrix.size(1), 2)
          merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
          label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

          out, mask = model2(merged_matrix, model2.training, 0)
          train1_score_tmp = torch.zeros(label_sbj.shape, device=device)
          sys_score_tmp = torch.zeros(label_sbj.shape, device=device)

          for mask_index in range(regress_weight_tr_repeat.shape[1]):
              out, mask = model2(merged_matrix, model2.training, mask_index*2)
              train1_score_tmp[:,mask_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], merged_matrix[:, mask_index * 2, :].T)[:, 0]
              sys_score_tmp[:,mask_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], out.logits[:,mask_index * 2, 0, :].T)[:, 0]
              # np.corrcoef(label_sbj[:, 0].detach().cpu().numpy(), train1_score_tmp[:, 0].detach().cpu().numpy())

          train1_score = torch.cat((train1_score, torch.tensor(train1_score_tmp, device=device_cpu)), dim=0)
          sys_score = torch.cat((sys_score, torch.tensor(sys_score_tmp, device=device_cpu)), dim=0)



    for i in range(label_train1_corr.shape[1]):
        label_train1_corr[0,i] = np.corrcoef(label_score[1:,i].detach().cpu().numpy(), train1_score[1:,i].detach().cpu().numpy())[0,1]
        label_sys_corr[0, i] = np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]
        train1_sys_corr[0, i] = np.corrcoef(train1_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]



    # if pred_score.shape[0] != 2586:
    #     print(correct[0])
    #     np.save('vit_aug_sink_lossop_' +str(epoch)+ '.npy', np.array(correct))

    # label_score
    # correct = np.sum(correct)/pred_score.shape[1]
    # regress_weight = torch.squeeze(torch.stack(regress_weight), axis=1)
    # regress_weight1 = regress_weight.cpu()
    # del label_score

    return label_train1_corr.mean(),  label_sys_corr.mean(), train1_sys_corr.mean()#correct,  correct, fusion_feature[1:,:,:],  regress_weight1 #/ len(loader.dataset)


def test_acc_zero(train_loader_stage2):
    model2.eval()
    correct = []
    device_cpu = torch.device('cpu')

    with torch.no_grad():
      label_sys_corr =  torch.tensor(torch.zeros(1, (text_feature_all.shape[1]-text_feature.shape[1])), device=device_cpu)
      label_score  =  torch.tensor(torch.zeros(1, (text_feature_all.shape[1]-text_feature.shape[1])), device=device_cpu)
      sys_score =   torch.tensor(torch.zeros(1, (text_feature_all.shape[1]-text_feature.shape[1])), device=device_cpu)

      for data in train_loader_stage2:
          data = data.to(device)
          optimizer2.zero_grad()

          fusion_feature_sbj = data.x.view(opt.batchSize, 28, 1024)
          label_sbj = data.y.view(opt.batchSize, 51)


          #28
          label_sbj = label_sbj[:,28:]
          sys_score_tmp = torch.zeros(label_sbj.shape, device=device)

          # mask  6:7 read_unage-->read_age
          for mask_index in range(label_sbj.shape[1]):
              fusion_feature_sbj[:, 0:1, 512:] = text_feature_zero[:, mask_index:mask_index+1, :]
              merged_matrix = torch.cat([fusion_feature_sbj, regress_weight_tr_repeat], dim=1)
              odd_columns = torch.arange(1, merged_matrix.size(1), 2)
              merged_matrix[:, odd_columns, :] = fusion_feature_sbj
              even_columns = torch.arange(0, merged_matrix.size(1), 2)
              merged_matrix[:, even_columns, :] = regress_weight_tr_repeat
              # label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

              out, mask = model2(merged_matrix, model2.training, mask_index * 2)
              sys_score_tmp[:,mask_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], out.logits[:,mask_index * 2, 0, :].T)[:, 0]


              # mask_index=6
              # out, mask = model2(merged_matrix, model2.training, mask_index*2)

          sys_score = torch.cat((sys_score, torch.tensor(sys_score_tmp, device=device_cpu)), dim=0)
          label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

      for i in range(label_sys_corr.shape[1]):
              label_sys_corr[0, i] = \
              np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]


    out_cor = torch.mean(torch.abs(label_sys_corr))
    return out_cor, label_sys_corr


############################   Model Training #########################################
best_loss = 1e10
val_acc_list = []


fusion_feature = fusion_feature_tr
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
train_dataset_stage2 = CustomDataset(fusion_feature, label_score_tr)
train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)
# torch.matmul(fusion_feature_tr[0,0,:].T, regress_weight_tr[0,:])
fusion_feature = fusion_feature_val
fusion_feature = fusion_feature.view(fusion_feature.shape[0], fusion_feature.shape[1]*fusion_feature.shape[2])
val_dataset_stage2 = CustomDataset(fusion_feature, label_score_val)
val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=True, drop_last=True)


regress_weight_tr_repeat = regress_weight_tr.unsqueeze(0).repeat(opt.batchSize, 1, 1).to(device)

#####################################################################


for epoch in range(stage1_epoch, 1000):
    since  = time.time()

    tr_loss=  train2(epoch, train_loader_stage2)
    label_train1_corr_mean,  label_sys_corr_mean, train1_sys_corr_mean = test_acc2(val_dataset_stage2)
    label_sys_corr_train_mean, _= test_acc_zero(train_loader_stage2)    #   test_acc2(val_dataset_stage2)
    label_sys_corr_val_mean, _= test_acc_zero(val_dataset_stage2)    #   test_acc2(val_dataset_stage2)



    time_elapsed = time.time() - since


    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '.format(epoch, tr_loss))



    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, label_train1_corr_mean: {:.7f}, label_sys_corr_mean: {:.7f}, train1_sys_corr_mean{:.7f}'.format(epoch, tr_loss, label_train1_corr_mean,  label_sys_corr_mean, train1_sys_corr_mean))

    print(label_sys_corr_train_mean, label_sys_corr_val_mean)
    print('*====**')




