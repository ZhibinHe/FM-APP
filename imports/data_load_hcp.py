import os
from imports.ABIDEDataset import HCPfmriScoreDataset_sbjnum, HCPT1wScoreDataset_sbjnum, HCPAfmriScoreDataset_sbjnum, HCPAT1wScoreDataset_sbjnum
from torch_geometric.data import DataLoader
from imports.utils import train_val_test_split_hcp
from net.Network_Combine import *
from net.models_mae import *
import pandas as pd
import torch


def data_load_hcp(path, name, opt):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_tr.pt'))
    ##################
    text_feature2 = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_te.pt'))
    text_feature = torch.cat((text_feature, text_feature2), dim=1)
########################
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    # dataset2 = HCPT1wScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900 Release Subjects 4rsfmri.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_all_phenotype.csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    csv_fname_values = csvdata["rsfmrilist"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (3427, 333, 333))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1



def data_load_hcp_t1fmri(path, path2, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_tr.pt'), map_location=device)

    text_feature2 = torch.load(os.path.join(opt.datadir + '/phenotype_text_feature_te.pt'), map_location=device)
    text_feature = torch.cat((text_feature, text_feature2), dim=1)

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    dataset2 = HCPT1wScoreDataset_sbjnum(path2, name)

    multi_modal_data = torch.zeros(dataset.data.y.size()[0],dataset.data.x.size()[1],dataset.data.x.size()[1]+dataset2.data.x.size()[1])
    fmri_data = torch.reshape(dataset.data.x, (dataset.data.y.size()[0], dataset.data.x.size()[1],dataset.data.x.size()[1]))
    t1w_data = torch.reshape(dataset2.data.x, (dataset2.data.y.size()[0], dataset.data.x.size()[1],dataset2.data.x.size()[1]))

    delete_index = []

    multi_modal_data[:, :, 0:fmri_data.size()[1]] = fmri_data
    for i in range(fmri_data.size()[0]):
        try:
            my_index = dataset2.data.sbj_fname.index(dataset.data.sbj_fname[i][0:6])
            multi_modal_data[i, :, fmri_data.size()[1]:] = t1w_data[my_index, :, :]
        except:
            delete_index.append(i)
            # print(dataset.data.sbj_fname[i])

    fmri_data =  multi_modal_data.index_select(0, torch.tensor([i for i in range(multi_modal_data.size(0)) if i not in torch.tensor(delete_index)]))


    dataset.data.x = torch.reshape(fmri_data, (fmri_data.size(0) * fmri_data.size(1), 333+9))
    dataset.data.sbj_fname = [row for i, row in enumerate(dataset.data.sbj_fname) if i not in delete_index]



    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/S900 Release Subjects 4rsfmrit1w.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_all_phenotype.csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    # select_score = np.delete(select_score, delete_index, axis=1)

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    # csvdata["rsfmrilist"] = csvdata["rsfmrilist"].drop(csvdata["rsfmrilist"].index[delete_index])



    csv_fname_values = csvdata["rsfmrilist"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (dataset_sbj_fname.size, 333, 333+9))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333+9))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1



def data_load_hcpa_t1fmri(path, path2, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold

    text_feature = torch.load(os.path.join(opt.datadir + '/HCPA_phenotype_text_feature_tr.pt'), map_location=device)
    ##################
    text_feature2 = torch.load(os.path.join(opt.datadir + '/HCPA_phenotype_text_feature_te.pt'), map_location=device)
    text_feature = torch.cat((text_feature, text_feature2), dim=1)
########################
    dataset = HCPAfmriScoreDataset_sbjnum(path, name)
    dataset2 = HCPAT1wScoreDataset_sbjnum(path2, name)

    multi_modal_data = torch.zeros(dataset.data.y.size()[0],dataset.data.x.size()[1],dataset.data.x.size()[1]+dataset2.data.x.size()[1])
    fmri_data = torch.reshape(dataset.data.x, (dataset.data.y.size()[0], dataset.data.x.size()[1],dataset.data.x.size()[1]))
    t1w_data = torch.reshape(dataset2.data.x, (dataset2.data.y.size()[0], dataset.data.x.size()[1],dataset2.data.x.size()[1]))

    delete_index = []

    multi_modal_data[:, :, 0:fmri_data.size()[1]] = fmri_data
    for i in range(fmri_data.size()[0]):
        try:
            my_index = dataset2.data.sbj_fname.index(dataset.data.sbj_fname[i][:-5])
            multi_modal_data[i, :, fmri_data.size()[1]:] = t1w_data[my_index, :, :]
        except:
            delete_index.append(i)
            # print(dataset.data.sbj_fname[i])

    fmri_data =  multi_modal_data.index_select(0, torch.tensor([i for i in range(multi_modal_data.size(0)) if i not in torch.tensor(delete_index)]))


    dataset.data.x = torch.reshape(fmri_data, (fmri_data.size(0) * fmri_data.size(1), 333+9))
    dataset.data.sbj_fname = [row for i, row in enumerate(dataset.data.sbj_fname) if i not in delete_index]



    csvdata = pd.read_csv(os.path.join(opt.csvroot + '/HCP_Aging_phenotype-select.csv'))

    # traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCP_train_phenotype.csv'))
    traincsvdata = pd.read_csv(os.path.join(opt.datadir + '/HCPA_all_phenotype .csv'))

    select_score = np.zeros((traincsvdata['0'].shape[0], csvdata.shape[0]))

    for i in range(traincsvdata['0'].shape[0]):
        select_score[i] = csvdata[traincsvdata['0'][i]]

    # select_score = np.delete(select_score, delete_index, axis=1)

    non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    select_score_1 = select_score[:, non_nan_columns]

    # csvdata["rsfmrilist"] = csvdata["rsfmrilist"].drop(csvdata["rsfmrilist"].index[delete_index])



    csv_fname_values = csvdata["subject_id_fmri"][non_nan_columns].values

    dataset.data.sbj_fname = [str(x) for x in dataset.data.sbj_fname]

    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    ###score
    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    y_arr = (select_score_2 - np.mean(select_score_2, axis=1, keepdims=True)) / np.std(select_score_2, axis=1,
                                                                                       keepdims=True)
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])
    dataset_x = np.reshape(dataset.data.x, (dataset_sbj_fname.size, 333, 333+9))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 333+9))

    #########################
    dataset.data.x = dataset_x
    dataset.data.y = y_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch

    dataset.data.x[dataset.data.x == float('inf')] = 0

    for i in range(select_fname.size):
        select_fname[i] = select_fname[i][:-5]
    select_fname2 = np.unique(select_fname)
    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=select_fname2.size, fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # test_index1 = list()
    train_index1 = list()
    val_index1 = list()

    for tr in tr_index:
        train_index1 = np.concatenate((train_index1, np.where(select_fname2[tr] == select_fname)[0]))
    train_index1 = train_index1.astype(np.int64)

    for val in val_index:
        val_index1 = np.concatenate((val_index1, np.where(select_fname2[val] == select_fname)[0]))
    val_index1 = val_index1.astype(np.int64)



    train_dataset = dataset[train_index1]
    val_dataset = dataset[val_index1]
    test_dataset = dataset[val_index1]


    return train_dataset, val_dataset, test_dataset, text_feature, dataset, train_index1, val_index1
