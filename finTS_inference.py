# 使用多个pretrained网络作为teachers，训练较浅层的student，训练权重，学习特征层。
# 添加一个重构网络，用来指导多个teachers的权重、使得student学到正确的特征。

import torch
from torch import nn
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import manifold
import MY_Data
import argparse


class Weight_Class(nn.Module):
    def __init__(self, n):
        super(Weight_Class, self).__init__()
        self.n = n
        self.w = nn.parameter.Parameter(torch.zeros([n, 1]))

    def forward(self, input):
        self.weight = nn.functional.softmax(self.w, dim=0)
        output = input * self.weight
        output = torch.sum(output, dim=1)
        return output


class Student_Class(nn.Module):
    def __init__(self):
        super(Student_Class, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=False)

    def forward(self, input):
        output = self.net(input)
        return output


class Student_Class_vgg(nn.Module):
    def __init__(self):
        super(Student_Class_vgg, self).__init__()
        self.net = torchvision.models.vgg11(pretrained=False)

    def forward(self, input):
        output = self.net(input)
        return output


class Student_Class_densenet(nn.Module):
    def __init__(self):
        super(Student_Class_densenet, self).__init__()
        self.net = torchvision.models.densenet121(pretrained=False)

    def forward(self, input):
        output = self.net(input)
        return output


class upsample(nn.Module):
    def __init__(self, in_c, out_c, activation):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=in_c),
            activation
        )
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=in_c)
        self.act1 = activation
        self.conv2 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=out_c)
        self.act2 = activation

    def forward(self, input):
        up = self.up(input)
        output = self.conv1(up)
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.act1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act2(output)
        return output


class Student_de_class(nn.Module):
    def __init__(self):
        super(Student_de_class, self).__init__()
        self.linear1 = nn.Linear(in_features=1000, out_features=6272)
        self.up1 = upsample(in_c=128, out_c=64, activation=nn.LeakyReLU(0.2))
        self.up2 = upsample(in_c=64, out_c=32, activation=nn.LeakyReLU(0.2))
        self.up3 = upsample(in_c=32, out_c=16, activation=nn.LeakyReLU(0.2))
        self.up4 = upsample(in_c=16, out_c=8, activation=nn.LeakyReLU(0.2))
        self.up5 = upsample(in_c=8, out_c=3, activation=nn.LeakyReLU(0.2))
        # self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.linear1(input)
        output = torch.reshape(output, [-1, 128, 7, 7])
        output = self.up1(output)
        output = self.up2(output)
        output = self.up3(output)
        output = self.up4(output)
        output = self.up5(output)
        # output = self.tanh(output)
        return output


def normalization(score):
    return (score - score.min()) / (score.max() - score.min())


def plt_patch(image, nor=True):
    imgplt = image.detach().numpy()
    if nor:
        imgplt = normalization(imgplt)
    for iplt in range(imgplt.shape[0]):
        plt.subplot(int(np.sqrt(imgplt.shape[0])), int(np.sqrt(imgplt.shape[0])), iplt + 1)
        plt.imshow(imgplt[iplt].transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'cifar10'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--evalepoch', type=int, default=50)
    parser.add_argument('--filepath', type=str, default='./run/exp')
    parser.add_argument('--cover', action='store_true')
    parser.add_argument('--student', type=str, default='resnet', choices=['resnet', 'vgg', 'densenet'])
    parser.add_argument('--savemodel', action='store_true')
    opt = parser.parse_args()


    if not os.path.exists(opt.filepath):
        print(opt.filepath, 'is not exist.')
        os._exit(0)



    print('Dataset:', opt.dataset, 'Epoch:', opt.epoch, 'Batch_size(CIFAR10):', opt.batch_size, 'Eval_step:',
          opt.evalepoch, 'Save_model_and_result_to:', opt.filepath)
    # raise ValueError('my error stop')

    cuda_availabel = torch.cuda.is_available()
    if cuda_availabel:
        maplocation = torch.device('cuda')
    else:
        maplocation = torch.device('cpu')

    # torchvision.models.resnet18(pretrained=True)
    # torchvision.models.vgg11(pretrained=True)
    # torchvision.models.densenet121(pretrained=True)

    Teacher_1 = torchvision.models.resnet152(pretrained=True).eval()
    # Teacher_1 = create_feature_extractor(Teacher_1, ['flatten'])# 2048
    Teacher_2 = torchvision.models.vgg19(pretrained=True).eval()
    # Teacher_2 = create_feature_extractor(Teacher_2, ['classifier.3'])# 4096
    Teacher_3 = torchvision.models.densenet201(pretrained=True).eval()
    # Teacher_3 = create_feature_extractor(Teacher_3, ['flatten'])# 1920


    if cuda_availabel:
        Teacher_1 = Teacher_1.cuda()
        Teacher_2 = Teacher_2.cuda()
        Teacher_3 = Teacher_3.cuda()
    batch_size = opt.batch_size
    epoch = opt.epoch

    result = list()
    res = list()
    for normal_lab in range(opt.start, 15 if opt.dataset == 'mvtec' else 10):
        bestAUC = 0
        if opt.dataset == 'cifar10':
            myCIFAR10 = MY_Data.CIFAR10()
            [data_loader, data_loader_test, label_test] = myCIFAR10.get(normal_lab=normal_lab, batch_size=batch_size)
        if opt.dataset == 'mvtec':
            myMVTec = MY_Data.MVTec()
            [data_loader, data_loader_test, label_test] = myMVTec.get_1024(normal_lab=normal_lab, batch_size=4)

        # if opt.student == 'resnet':
        #     Net_SEn = Student_Class()
        # elif opt.student == 'vgg':
        #     Net_SEn = Student_Class_vgg()
        # elif opt.student == 'densenet':
        #     Net_SEn = Student_Class_densenet()
        # else:
        #     raise ValueError('Error student model!!!')
        # Net_SDe = Student_de_class()
        # Net_Weight = Weight_Class(3)
        # Opt_SEn = torch.optim.Adam(Net_SEn.parameters())
        # Opt_SDe = torch.optim.Adam(Net_SDe.parameters())
        # Opt_Weight = torch.optim.Adam(Net_Weight.parameters())

        MSE = torch.nn.MSELoss()
        CosSim = torch.nn.CosineSimilarity()
        CE = torch.nn.CrossEntropyLoss()
        if cuda_availabel:
            # Net_SEn = Net_SEn.cuda()
            # Net_SDe = Net_SDe.cuda()
            # Net_Weight = Net_Weight.cuda()
            MSE = MSE.cuda()
            CosSim = CosSim.cuda()
            CE = CE.cuda()

        # compute mean and std for all training data
        try:
            if cuda_availabel:
                [y_normal_T1_mean, y_normal_T2_mean, y_normal_T3_mean,
                 y_normal_T1_std, y_normal_T2_std, y_normal_T3_std] = torch.load(
                    opt.filepath + '/meanstd' + str(normal_lab) + '.data', map_location=torch.device('cuda'))
            else:
                [y_normal_T1_mean, y_normal_T2_mean, y_normal_T3_mean,
                 y_normal_T1_std, y_normal_T2_std, y_normal_T3_std] = torch.load(
                    opt.filepath + '/meanstd' + str(normal_lab) + '.data', map_location=torch.device('cpu'))
            print('load mean std from file: ', opt.filepath + '/meanstd' + str(normal_lab) + '.data')
        except:
            y_normal_T1 = list()
            y_normal_T2 = list()
            y_normal_T3 = list()
            for i, (image, _) in enumerate(data_loader):
                if opt.dataset == 'mvtec':
                    image = myMVTec.Fullcut(image)
                    image = torch.cat(image)
                if cuda_availabel:
                    image = image.cuda()
                with torch.no_grad():
                    y1 = Teacher_1(image)
                    y2 = Teacher_2(image)
                    y3 = Teacher_3(image)
                y_normal_T1.append(y1)
                y_normal_T2.append(y2)
                y_normal_T3.append(y3)
                print('', end='\r')
                print('Compute Teachers\'s output means and stds on training dataset,', i + 1, '/', len(data_loader),
                      end='')
            y_normal_T1 = torch.cat(y_normal_T1)
            y_normal_T1_mean = torch.mean(y_normal_T1, dim=0)
            y_normal_T1_std = torch.std(y_normal_T1, dim=0)
            y_normal_T1_std[y_normal_T1_std == 0] = 1e-3
            y_normal_T2 = torch.cat(y_normal_T2)
            y_normal_T2_mean = torch.mean(y_normal_T2, dim=0)
            y_normal_T2_std = torch.std(y_normal_T2, dim=0)
            y_normal_T2_std[y_normal_T2_std == 0] = 1e-3
            y_normal_T3 = torch.cat(y_normal_T3)
            y_normal_T3_mean = torch.mean(y_normal_T3, dim=0)
            y_normal_T3_std = torch.std(y_normal_T3, dim=0)
            y_normal_T3_std[y_normal_T3_std == 0] = 1e-3
            torch.save([y_normal_T1_mean, y_normal_T2_mean, y_normal_T3_mean,
                        y_normal_T1_std, y_normal_T2_std, y_normal_T3_std],
                       opt.filepath + '/meanstd' + str(normal_lab) + '.data')



        # 训练一定次数验证一次，保留最好的model和anomalyscore

        Net_SEn = torch.load(opt.filepath + '/Net_SEn' + str(normal_lab) + '.pkl', map_location=maplocation).eval()
        Net_SDe = torch.load(opt.filepath + '/Net_SDe' + str(normal_lab) + '.pkl', map_location=maplocation).eval()
        Net_Weight = torch.load(opt.filepath + '/Net_Weight' + str(normal_lab) + '.pkl',map_location=maplocation).eval()


        # 用teachers和student的差值+student的重构误差作为anomaly score
        Net_SEn.eval()
        Net_SDe.eval()
        Net_Weight.eval()

        # 计算用于nor的max和min mu std
        AS_z = list()
        AS_cossim = list()
        AS_rec = list()
        for i, (image, _) in enumerate(data_loader):
            if opt.dataset == 'mvtec':
                image = myMVTec.Fullcut(image)
                image = torch.cat(image)
            if cuda_availabel:
                image = image.cuda()
            with torch.no_grad():
                z_S = Net_SEn(image)
                x_S = Net_SDe(z_S)
                z1 = (Teacher_1(image) - y_normal_T1_mean) / y_normal_T1_std
                z2 = (Teacher_2(image) - y_normal_T2_mean) / y_normal_T2_std
                z3 = (Teacher_3(image) - y_normal_T3_mean) / y_normal_T3_std
                z_T = Net_Weight(torch.stack([z1, z2, z3], dim=1))
                x_T = Net_SDe(z_T)
            AS_z.append(torch.mean((z_S - z_T) ** 2, dim=1))
            AS_cossim.append(1 - CosSim(z_S, z_T))
            AS_rec.append(torch.mean((x_S - image) ** 2, dim=[1, 2, 3]))
            print('', end='\r')
            print('Compute max min,', i + 1, '/', len(data_loader), end='')
        AS_z = torch.cat(AS_z).cpu()
        AS_cossim = torch.cat(AS_cossim).cpu()
        AS_rec = torch.cat(AS_rec).cpu()
        AS_z_max = AS_z.max()
        AS_z_min = AS_z.min()
        # AS_z_mu = AS_z.mean()
        # AS_z_std = AS_z.std()

        AS_cossim_max = AS_cossim.max()
        AS_cossim_min = AS_cossim.min()
        # AS_cossim_mu = AS_cossim.mean()
        # AS_cossim_std = AS_cossim.std()

        AS_rec_max = AS_rec.max()
        AS_rec_min = AS_rec.min()
        # AS_rec_mu = AS_rec.mean()
        # AS_rec_std = AS_rec.std()



        AS_z = list()
        AS_cossim = list()
        AS_rec = list()
        for i, (image, _) in enumerate(data_loader_test):
            if opt.dataset == 'mvtec':
                image = myMVTec.Fullcut(image)
                image = torch.cat(image)
            if cuda_availabel:
                image = image.cuda()
            with torch.no_grad():
                z_S = Net_SEn(image)
                x_S = Net_SDe(z_S)
                z1 = (Teacher_1(image) - y_normal_T1_mean) / y_normal_T1_std
                z2 = (Teacher_2(image) - y_normal_T2_mean) / y_normal_T2_std
                z3 = (Teacher_3(image) - y_normal_T3_mean) / y_normal_T3_std
                z_T = Net_Weight(torch.stack([z1, z2, z3], dim=1))
                x_T = Net_SDe(z_T)
            AS_z.append(torch.mean((z_S - z_T) ** 2, dim=1))
            AS_cossim.append(1 - CosSim(z_S, z_T))
            AS_rec.append(torch.mean((x_S - image) ** 2, dim=[1, 2, 3]))
            print('', end='\r')
            print('Compute Anomaly Score,', i + 1, '/', len(data_loader_test), end='')
        AS_z = torch.cat(AS_z).cpu()
        AS_cossim = torch.cat(AS_cossim).cpu()
        AS_rec = torch.cat(AS_rec).cpu()

        torch.save([AS_z, AS_cossim, AS_rec], opt.filepath + '/anomaly_score' + str(normal_lab) + '.data')

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
        # columns.append('normalization')
        # list_as.append(normalization(AS_z) + normalization(AS_cossim) + normalization(AS_rec))
        columns.append('nor')
        list_as.append((AS_z-AS_z_min)/(AS_z_max-AS_z_min) + (AS_cossim-AS_cossim_min)/(AS_cossim_max-AS_cossim_min) + (AS_rec-AS_rec_min)/(AS_rec_max-AS_rec_min))
        # columns.append('std')
        # list_as.append((AS_z-AS_z_mu)/AS_z_std + (AS_cossim-AS_cossim_mu)/AS_cossim_std + (AS_rec-AS_rec_mu)/AS_rec_std)

        for Anomaly_Score in list_as:
            Anomaly_Score = Anomaly_Score.reshape(label_test.size, -1)
            Anomaly_Score = torch.max(Anomaly_Score, dim=1).values
            AUC = roc_auc_score(y_true=label_test, y_score=Anomaly_Score)
            res_class.append(AUC)
        res.append(np.stack(res_class))
        print(res_class)
    res = np.stack(res)

    res_mean = res.mean(axis=0)
    res_mean[0] = 0
    res_mean.max()
    print(res_mean)
    print(res_mean.argmax(), columns[res_mean.argmax()], res_mean.max())
    # res = np.concatenate([res, res_mean[np.newaxis, :]], axis=0)
    pd.DataFrame(columns=columns, data=res).to_csv(opt.filepath + '/Eval_' + opt.student + '.csv', encoding='utf-8')



    #
    #
    #
    #
    #
    #     res_class = list()
    #     # res_class.append(normal_lab)
    #     list_as = [AS_z, AS_cossim, AS_rec]
    #     columns = ['normal_lab', 'AS_z', 'AS_cossim', 'AS_rec']
    #     for r1 in torch.linspace(0, 1, 11):
    #         for r2 in torch.linspace(0, (1 - r1), int((1 - r1) / 0.1 + 1)):
    #             r3 = 1 - r1 - r2
    #             list_as.append(r1 * AS_z + r2 * AS_cossim + r3 * AS_rec)
    #             # 改进 list_as.append(r1 * normalization(AS_z) + r2 * normalization(AS_cossim) + r3 * normalization(AS_rec))
    #             columns.append(str(r1) + str(r2) + str(r3))
    #     for Anomaly_Score in list_as:
    #         Anomaly_Score = Anomaly_Score.reshape(label_test.size, -1)
    #         Anomaly_Score = torch.max(Anomaly_Score, dim=1).values
    #         AUC = roc_auc_score(y_true=label_test, y_score=Anomaly_Score)
    #         res_class.append(AUC)
    #     res_class = np.stack(res_class)
    #
    #     print('max AUC: {}, for class {}, epoch: {}'.format(res_class.max(), normal_lab, e + 1))
    #     result.append([normal_lab, e + 1, res_class.max()])
    #     pd.DataFrame(columns=['Normal Class', 'Epoch', 'AUC'], data=result).to_csv(
    #         opt.filepath + '/finTS_' + opt.student + '.csv', encoding='utf-8')
    #
    #     torch.save([AS_z, AS_cossim, AS_rec], opt.filepath + '/anomaly_score' + str(normal_lab) + '.data')
    #
    #
    #
    # # compute results on different anomaly score, find max average AUC.
    # res = list()
    # for normal_lab in range(15 if opt.dataset == 'mvtec' else 10):
    #     if opt.dataset == 'cifar10':
    #         myCIFAR10 = MY_Data.CIFAR10()
    #         [_, data_loader_test, label_test] = myCIFAR10.get(normal_lab=normal_lab, batch_size=1)
    #     if opt.dataset == 'mvtec':
    #         myMVTec = MY_Data.MVTec()
    #         [_, data_loader_test, label_test] = myMVTec.get_1024(normal_lab=normal_lab, batch_size=1)
    #     AS_z, AS_cossim, AS_rec = torch.load(opt.filepath + '/anomaly_score' + str(normal_lab) + '.data',
    #                                          map_location='cpu')
    #     res_class = list()
    #     res_class.append(normal_lab)
    #     list_as = [AS_z, AS_cossim, AS_rec]
    #     columns = ['normal_lab', 'AS_z', 'AS_cossim', 'AS_rec']
    #     for r1 in torch.linspace(0, 1, 11):
    #         for r2 in torch.linspace(0, (1 - r1), int((1 - r1) / 0.1 + 1)):
    #             r3 = 1 - r1 - r2
    #             list_as.append(r1 * AS_z + r2 * AS_cossim + r3 * AS_rec)
    #             # 改进 list_as.append(r1 * normalization(AS_z) + r2 * normalization(AS_cossim) + r3 * normalization(AS_rec))
    #             columns.append(str(r1) + str(r2) + str(r3))
    #     for Anomaly_Score in list_as:
    #         Anomaly_Score = Anomaly_Score.reshape(label_test.size, -1)
    #         Anomaly_Score = torch.max(Anomaly_Score, dim=1).values
    #         AUC = roc_auc_score(y_true=label_test, y_score=Anomaly_Score)
    #         res_class.append(AUC)
    #     res.append(np.stack(res_class))
    # res = np.stack(res)
    #
    # res_mean = res.mean(axis=0)
    # res_mean[0] = 0
    # res_mean.max()
    # print(res_mean)
    # print(res_mean.argmax(), columns[res_mean.argmax()], res_mean.max())
    # res = np.concatenate([res, res_mean[np.newaxis, :]], axis=0)
    # pd.DataFrame(columns=columns, data=res).to_csv(opt.filepath + '/Eval_' + opt.student + '.csv', encoding='utf-8')
