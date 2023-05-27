import numpy as np
import pandas as pd
import time

start = time.time()

# 读取数据


target = np.random.randint(1, 31, size=(1000))
data = np.random.random((1000, 10))


class dbscan(object):
    def __init__(self, train_data, train_labels):
        self.tdata = train_data
        self.tlabels = train_labels

    def predict_test(self, test_data, test_label):
        predict_test_label = []
        datalabel, Eps = self.reshape_label(self.tlabels, self.tdata)
        for test_item in test_data:
            dis = (self.tdata - test_item) ** 2
            dis = np.sum(dis, axis=1)
            item_neighbors = np.where(dis[:] <= Eps)[0]
            tem_list2 = [0] * 31
            for test_neighbor in item_neighbors:
                tem_list2[datalabel[test_neighbor]-1] += 1
            if max(tem_list2) >= len(test_data) / 20:
                predict_test_label.append(tem_list2.index(max(tem_list2))+1)
            else:
                predict_test_label.append(tem_list2.index(max(tem_list2))+1)
            print(predict_test_label)
        return predict_test_label, Eps

    def accuracy(self, test_data, test_label):
        pre, Eps = self.predict_test(test_data, test_label)
        right_num = 0
        for i in range(len(test_data)):
            if pre[i] == test_label[i]:
                right_num += 1
        accuracy = right_num / len(pre)
        return accuracy, Eps

    def regionQuery(self, p, dis, Eps):
        """
        返回点p的密度直达点
        """
        neighbors = np.where(dis[:, p] <= Eps ** 2)[0]
        return neighbors

    def growCluster(self, dis, pre_target, labels, p, Eps, MinPts):
        """
        寻找p点的所有密度可达点，形成最终一个簇
        输入：距离矩阵、预测标签、初始点p、是否被遍历过的标签、邻域半径、邻域中数据对象数目阈值
        """

        # 如果该点已经经过遍历，结束对该点的操作
        if labels[p] == -1:
            return labels, pre_target

        # p的密度直达点
        NeighborPts = self.regionQuery(p, dis, Eps)

        # 遍历p的密度直达点
        i = 0
        while i < len(NeighborPts):
            Pn = NeighborPts[i]
            # 找出Pn的密度直达点
            PnNeighborPts = self.regionQuery(Pn, dis, Eps)
            # 如果此时的点是核心点
            if len(PnNeighborPts) >= MinPts:
                # 将点Pn的新的密度直达点加入点簇
                Setdiff1d = np.setdiff1d(PnNeighborPts, NeighborPts)  # 在PnNeighborPts不在NeighborPts中
                NeighborPts = np.hstack((NeighborPts, Setdiff1d))
            # 否则，说明为边界点，什么也不需要做
            # NeighborPts = NeighborPts
            i += 1

        # 将点p密度可达各点归入p所在簇
        pre_target[NeighborPts] = pre_target[p]
        labels[NeighborPts] = -1
        return labels, pre_target

    def DBSCAN(self, n, k, dis, Eps, MinPts, mode=2):
        """
        输入：距离矩阵、邻域半径、邻域中数据对象数目阈值
        输出：mode==1:预测值准确性（平均标准差），运行时间;mode==2:预测值
        """
        temp_start = int(round(time.time() * 1000000))

        p = 0
        labels = np.zeros(n)  # 有两个可能的值：-1：完成遍历的；0：这个点还没经历过遍历，初始均为0
        pre_target = np.arange(n)

        # 从第一个点开始遍历
        while p < n:
            # 寻找当前点的密度可达点，形成一个簇
            labels, pre_target = self.growCluster(dis, pre_target, labels, p, Eps, MinPts)
            # 此时的簇数
            c_num = len(np.unique(pre_target))
            # if mode == 2:
            # print("循环迭代次数：{}，此时有{}个簇".format(p + 1, c_num))
            # 分成小于k簇直接跳出循环（说明分得有问题）
            # 分成正好k簇也跳出循环，直接去检查有没有分对
            class_num = 0
            for ui in np.unique(pre_target):
                num = 0
                for uj in pre_target:
                    if ui == uj :
                        num+=1
                if num>=40:
                    class_num+=1
            if c_num <= k:
                break
            p += 1

        return pre_target, class_num, labels

    ######### 以上是重中之重 #########

    # 经过观察，Eps=4.0,MinPts=29可作为参数传入，
    # 准确率100%
    # 再次提示，测试、参数调整过程及可视化所用相关在文末完整项目中提供
    # pre_target = DBSCAN(n=n, k=k, dis=dis, Eps=4.0, MinPts=29, mode=1)
    def class_gesture(self, data, Eps=5.3, k=31, bian=0.1, MinPts=30):
        n = data.shape[0]
        # 初始化dis矩阵
        dis = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                dis[i][j] = np.sum((data[i] - data[j]) ** 2)
        # 求两两簇（点）之间的距离
        for i in range(n - 1):
            for j in range(i + 1, n):
                dis[j][i] = ((data[j] - data[i]) ** 2).sum()
        k_num = 0
        while True:
            pre_target, class_number, labelsone = self.DBSCAN(n=n, k=k, dis=dis, Eps=Eps, MinPts=MinPts)
            if class_number < 31:
                Eps = Eps - bian
            elif class_number == 31:
                break
            else:
                Eps += bian
            if k_num % 2 == 0:
                bian = 0.001
            if k_num % 2 == 1:
                bian = 0.01
            if k_num<10:
                bian = 1
            k_num += 1
            print(Eps)
        for i in range(len(labelsone)):
            if labelsone[i] == 0:
                pre_target[i] = 1000000
        print(pre_target)
        return pre_target, Eps

    def reshape_label(self, label, data):
        pre_label, Eps = self.class_gesture(data)
        pre_label_u = set(pre_label)
        for i_label in pre_label_u:
            tem_list = []
            tem_list1 = [0] * 32
            for label_index in range(len(pre_label)):
                if pre_label[label_index] == i_label:
                    tem_list.append(label_index)
            for item_label in tem_list:
                try:
                    tem_list1[label[item_label]] += 1
                except:
                    print(label(item_label))
            m_index = tem_list1.index(max(tem_list1))
            for item in range(len(pre_label)):
                if pre_label[item] == i_label:
                    pre_label[item] = m_index
        return pre_label, Eps


if __name__ == "__main__":
    train_labels = np.random.randint(1, 31, size=(10000))
    train_data = np.random.random((10000, 10))
    test_data = np.random.random((100, 10))
    test_label = np.random.randint(1, 31, size=(100))
    k = dbscan(train_data, train_labels, test_data, test_label)
    print(k.pre)
    print(k.acc)
    print(k.eps)
