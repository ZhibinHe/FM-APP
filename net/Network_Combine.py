import torch
import torch.nn as nn
from net.Network_Dual_ViT import *
from net.Network_Pred import *
from net.affinity_sink_layer import *
# from net.models_mae import *
import matplotlib.pyplot as plt
import pylab
import numpy as np

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        self.image_feature_extractor = Network_regress_score_512(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)


    def forward(self, fmri, text):
        # 前向传播过程
        features = self.image_feature_extractor(fmri)
        features1 = features.unsqueeze(1).expand(features.shape[0], text.shape[1], features.shape[1])
        text1 = text[1,:,:].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction = self.phenotype_prediction(combine_feature) #, regress_para

        return prediction #, regress_para

class CombinedModel_weight333(nn.Module):
    def __init__(self):
        super(CombinedModel_weight333, self).__init__()

        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)


    def forward(self, fmri, text):
        # 前向传播过程
        features = self.image_feature_extractor(fmri)

        features_mean = torch.mean(features, axis=1)


        features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para




class CombinedModel_weight(nn.Module):
    def __init__(self):
        super(CombinedModel_weight, self).__init__()

        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=332, fmri_outdim=1024, image_size=332,
                                                                 patch_size=332, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)


    def forward(self, fmri, text):
        # 前向传播过程
        features = self.image_feature_extractor(fmri)

        features_mean = torch.mean(features, axis=1)


        features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para


class CombinedModel_sink(nn.Module):
    def __init__(self):
        super(CombinedModel_sink, self).__init__()

        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=332, fmri_outdim=1024, image_size=332,
                                                                 patch_size=332, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para


class CombinedModel_sink333(nn.Module):
    def __init__(self):
        super(CombinedModel_sink333, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para

class CombinedModel_sink_feature(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature


class CombinedModel_sink_feature_hcpa(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_hcpa, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=24)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature



class CombinedModel_sink_feature_stage2(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_stage2, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.fc1 = torch.nn.Linear(9, 333)
        self.bn1 = torch.nn.BatchNorm1d(333)


    def forward(self, fmri):
        fmri = F.relu(self.fc1(fmri))


        fmri = self.bn1(fmri)

        features = self.image_feature_extractor(fmri)

        return features




class CombinedModel_sink_feature_t1w(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_t1w, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)   #depth=1

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))
        self.fc1 = torch.nn.Linear(9, 333)



    def forward(self, fmri, text):
        fmri = F.relu(self.fc1(fmri))

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature


class CombinedModel_sink_feature_t1w_24(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_t1w_24, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)   #depth=1

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=24)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))
        self.fc1 = torch.nn.Linear(9, 333)



    def forward(self, fmri, text):
        fmri = F.relu(self.fc1(fmri))

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature




class CombinedModel_sink_feature_t1w_23(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_t1w_23, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=23)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))
        self.fc1 = torch.nn.Linear(9, 333)



    def forward(self, fmri, text):
        fmri = F.relu(self.fc1(fmri))

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para, combine_feature


class CombinedModel_sink_feature_multimodel(nn.Module):
    def __init__(self):
        super(CombinedModel_sink_feature_multimodel, self).__init__()

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)

        self.affinityt1w = Affinity(512)
        self.instNormt1w = nn.InstanceNorm2d(1, affine=True)


        self.fc1 = torch.nn.Linear(9, 333)

        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri,t1w, text):

        features = self.image_feature_extractor(fmri)
        t1w = F.relu(self.fc1(t1w))
        features_t1w = self.image_feature_extractor(t1w)

        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)
        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        ###################t1w
        s_s1 = self.affinity(features_t1w, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features_t1w, 1, 2)
        features_mean = torch.matmul(features_t, s_s1)
        features1_t1w = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # combine_feature = torch.cat((features1, text1), axis=2)

        combine_feature_t1w = torch.cat((features1_t1w, text1), axis=2)



        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        prediction_t1w, regress_para_t1w = self.phenotype_prediction(combine_feature_t1w)


        return prediction , regress_para, combine_feature, prediction_t1w, regress_para_t1w, combine_feature_t1w




class CombinedModel_sink333_vitdepth(nn.Module):
    def __init__(self, depth=1):
        super(CombinedModel_sink333_vitdepth, self).__init__()

        self.depth = depth

        # self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
        #                                                          patch_size=333, num_classes=1024, dim=1024,
        #                                                          depth=1, heads=16, mlp_dim=2048)
        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=self.depth, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.affinity = Affinity(512)
        self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])




        s_s1 = self.affinity(features, text1)
        s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, s_s1)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para







class CombinedModel_cossim(nn.Module):
    def __init__(self):
        super(CombinedModel_cossim, self).__init__()

        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=332, fmri_outdim=1024, image_size=332,
                                                                 patch_size=332, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.linear_layer = nn.Linear(512, 512)

        # self.affinity = Affinity(512)
        # self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        features1 = self.linear_layer(features)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])



        similarity = F.cosine_similarity(features1.unsqueeze(2), text1.unsqueeze(1), dim=3)

        # 归一化
        # normalized_similarity = similarity / torch.sum(similarity, dim=1, keepdim=True)
        normalized_similarity = torch.exp(similarity)
        normalized_similarity = normalized_similarity / torch.sum(normalized_similarity, dim=1, keepdim=True)



        # s_s1 = self.affinity(features, text1)
        # s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        # log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        # s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, normalized_similarity)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para


class CombinedModel_cossim333(nn.Module):
    def __init__(self):
        super(CombinedModel_cossim333, self).__init__()

        self.image_feature_extractor = Network_regress_score_out333(fmri_indim=333, fmri_outdim=1024, image_size=333,
                                                                 patch_size=333, num_classes=1024, dim=1024,
                                                                 depth=1, heads=16, mlp_dim=2048)

        self.phenotype_prediction = MultiTaskRegressionNetwork(num_tasks=28)
        self.linear_layer = nn.Linear(512, 512)

        # self.affinity = Affinity(512)
        # self.instNorm = nn.InstanceNorm2d(1, affine=True)
        # self.task_weights = nn.Parameter(torch.ones(28))



    def forward(self, fmri, text):

        features = self.image_feature_extractor(fmri)
        features1 = self.linear_layer(features)
        text1 = text[1, :, :].unsqueeze(0).expand(features.shape[0], text.shape[1], features.shape[2])



        similarity = F.cosine_similarity(features1.unsqueeze(2), text1.unsqueeze(1), dim=3)

        # 归一化
        # normalized_similarity = similarity / torch.sum(similarity, dim=1, keepdim=True)
        normalized_similarity = torch.exp(similarity)
        normalized_similarity = normalized_similarity / torch.sum(normalized_similarity, dim=1, keepdim=True)



        # s_s1 = self.affinity(features, text1)
        # s_s1 = self.instNorm(s_s1[:, None, :, :]).squeeze(dim=1)
        # log_s_s1 = sinkhorn_rpm(s_s1, n_iters=100, slack=False)
        # s_s1 = torch.exp(log_s_s1)

        # plt.imshow(s_s1[0, :, :].detach().cpu().numpy())
        # pylab.show()
        # np.save("brain_sinkhore_active_map.npy", s_s1[0, :, :].detach().cpu().numpy())

        features_t = torch.transpose(features, 1, 2)


        features_mean = torch.matmul(features_t, normalized_similarity)
        features1 = torch.transpose(features_mean, 1, 2)

        # features_mean = torch.mean(features, axis=1)
        # features1 = features_mean.unsqueeze(1).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        # text1 = text[1,:,:].unsqueeze(0).expand(features_mean.shape[0], text.shape[1], features_mean.shape[1])
        combine_feature = torch.cat((features1, text1), axis=2)

        prediction , regress_para= self.phenotype_prediction(combine_feature) #
        return prediction , regress_para