#!/usr/bin/env python
# pylint: disable=W0201
import sys, os
import argparse
import yaml
import numpy as np
import zipfile
from random import sample
from os.path import basename
# torch
import torch
import torch.nn as nn
import torch.optim as optim
# Classify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from processor.Classify_tools.DBSCAN import dbscan
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        # self.model.apply(weights_init)
        self.KLloss = nn.KLDivLoss(reduction='batchmean')  # KL散度loss 
        self.loss = nn.MSELoss()  # MSEloss
        # self.loss = nn.L1Loss(reduction='mean')  # MAEloss

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_topk_knn(self, k, data, label):
        if k == 1:
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
            knn.fit(self.allmid, self.alllabel)
            score = knn.score(data, label)
        elif k == 5:
            top5count = []
            for i in range(k):
                knn = KNeighborsClassifier(n_neighbors=i + 1, algorithm='kd_tree')
                knn.fit(self.allmid, self.alllabel)
                result = knn.predict(data)
                top5count.append(result)
            right5count = 0
            for i in range(len(label)):
                if label[i] == top5count[0][i] or label[i] == top5count[1][i] or label[i] == top5count[2][i] or label[i] == top5count[3][i] or label[i] == top5count[4][i]:
                    right5count = right5count + 1
            score = right5count / len(label)

        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * score))
        
    def show_topk_RF(self, k, data, label):
        if k == 1:
            rf = RandomForestClassifier(random_state=1, n_estimators=100)
            rf.fit(self.allmid, self.alllabel)
            score = rf.score(data, label)
        elif k == 5:
            top5count = []
            for i in range(k):
                rf = RandomForestClassifier(random_state=i, n_estimators=50)
                rf.fit(self.allmid, self.alllabel)
                result = rf.predict(data)
                top5count.append(result)
            right5count = 0
            for i in range(len(label)):
                if label[i] == top5count[0][i] or label[i] == top5count[1][i] or label[i] == top5count[2][i] or label[i] == top5count[3][i] or label[i] == top5count[4][i]:
                    right5count = right5count + 1
            score = right5count / len(label)

        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * score))

    def show_topk_DBCSAN(self, k, data, label):
        if k == 1:
            db = dbscan(self.allmid, self.alllabel)
            score, _ = db.accuracy(data, label)
        elif k == 5:
            top5count = []
            for i in range(k):
                db = dbscan(self.allmid, self.alllabel)
                result, _ = db.predict_test(data, label)
                top5count.append(result)
            right5count = 0
            for i in range(len(label)):
                if label[i] == top5count[0][i] or label[i] == top5count[1][i] or label[i] == top5count[2][i] or label[i] == top5count[3][i] or label[i] == top5count[4][i]:
                    right5count = right5count + 1
            score = right5count / len(label)

        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * score))
        
    def create_submission(self, predPath):

        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath, 'Submission.csv')
        output_file = open(output_filename, 'w')

        # print(self.data_loader['test'].dataset.sample_name)
        # print(len(self.data_loader['test'].dataset.sample_name))
        for inx in range(len(self.data_loader['test'].dataset.sample_name)):
            output_file.write(str(self.data_loader['test'].dataset.sample_name[inx]) + "," + str(self.result[inx]) + "\n")
        output_file.close()

        # Set the name of the output zip file
        zip_file_name = os.path.join(predPath, 'Submission.zip')

        # Create a ZipFile object with the output zip file name and mode
        with zipfile.ZipFile(zip_file_name, "w") as zip_file:
            # Add the file you want to zip to the archive
            zip_file.write(output_filename, basename(output_filename))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        # print(len(loader))
        for data, label in loader:
            # print(data)
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            mid, output = self.model(data)
            N, C, T, V, M = data.size()
            data = data.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            data_bn = nn.BatchNorm1d(M * C * V)
            data_bn.cuda()
            data = data_bn(data)
            input = data.view(N * M, C * V, T).permute(0, 2, 1).contiguous()
            # input = data.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
            loss = self.loss(input, output) + 0.8 * self.KLloss(input.softmax(dim=-1).log(), output.softmax(dim=-1)) # 计算loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2) # 梯度截断
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        allsample = self.alldata['train']
        clasf = []
        alllabel = []
        for data, label in allsample:
            for i in label:
                alllabel.append(int(i))
            alld = data

            with torch.no_grad():
                allmid, _ = self.model(alld)
            allclassfi = allmid.view(allmid.shape[0], allmid.shape[1])
            allclassfi = allclassfi.cpu().detach().numpy()
            clasf.append(allclassfi)
        totaltclas = np.empty(shape=(403, 128))
        totaltlabel = np.array(alllabel)
        bsinprocesser = 32
        count = 0
        for i in clasf:
            totaltclas[count:count + bsinprocesser] = i
            count += bsinprocesser
        
        ####for half supervise
        '''countlb = {}
        for i in range(32):
            countlb[str(i)] = []
        for i in range(len(totaltlabel)):
            countlb[str(totaltlabel[i])].append(i)
        
        min_n = 600
        max_n = 1500
        partial_inx = []
        small = ['0', '1', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '20', '21', '24', '25', '26', '27', '29']
        big = ['30', '31']
        for item in countlb.keys():
            if item in small:
                new = countlb[item]
                partial_inx = partial_inx + new
            elif item in big:
                new = sample(countlb[item], max_n)
                partial_inx = partial_inx + new
            else:
                new = sample(countlb[item], min_n)
                partial_inx = partial_inx + new
        
        totaltclas = totaltclas.take(partial_inx, axis=0)
        totaltlabel = totaltlabel.take(partial_inx, axis=0)'''
        ####
        
        self.allmid = totaltclas
        self.alllabel = totaltlabel
        knn = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
        knn.fit(self.allmid, self.alllabel)
        # rf = RandomForestClassifier(random_state=1)
        # rf.fit(self.allmid, self.alllabel)
        # db = dbscan(self.allmid, self.alllabel)
        loss_value = []
        result_frag = []
        label_frag = []
        tclasf = []
        alltelabel = []
        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            for i in label:
                alltelabel.append(int(i))

            # inference
            with torch.no_grad():
                mid, output = self.model(data)

            classfi = mid.view(mid.shape[0], mid.shape[1])
            classfi = classfi.cpu().detach().numpy()
            tclasf.append(classfi)
            result = knn.predict(classfi)
            # result = rf.predict(classfi)
            # result, _ = db.predict_test(classfi, label)
            result_frag.append(result)

            # get loss
            if evaluation:
                N, C, T, V, M = data.size()
                data = data.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                data_bn = nn.BatchNorm1d(M * C * V)
                data_bn.cuda()
                data = data_bn(data)
                input = data.view(N * M, C * V, T).permute(0, 2, 1).contiguous()
                # input = data.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
                loss = self.loss(input, output) + 0.8 * self.KLloss(input.softmax(dim=-1).log(), output.softmax(dim=-1))  # 计算loss
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)

        totalteclasf = np.empty(shape=(207, 128))
        totaltelabel = np.array(alltelabel)
        tebsinprocesser = 32
        count = 0
        for i in tclasf:
            totalteclasf[count:count + bsinprocesser] = i
            count += tebsinprocesser

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk_knn(k, totalteclasf, totaltelabel)

            self.create_submission(self.arg.work_dir)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
