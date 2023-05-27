import netron
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.Unsupervise_tools import TemporalConvNet, AntiTemporalConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, num_person=1, in_channels=3, num_point=22, classifeature=100):
        super(EncoderTCN, self).__init__()
        self.in_channels = num_inputs
        self.channels = num_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.TCN = TemporalConvNet(num_inputs, num_channels, kernel_size)
        self.fcn1 = nn.Linear(in_channels * num_point * num_channels[-1], classifeature)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V * C, T).permute(0, 2, 1).contiguous()
        x = self.TCN(x)
        N, T, CV = x.size()
        x = x.view(N, T * CV)
        enout = self.fcn1(x)
        return enout


class DecoderATCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, in_channels=3, num_point=22, classifeature=100):
        super(DecoderATCN, self).__init__()
        self.in_channels = num_inputs
        self.channels = num_channels
        self.OT = num_channels[0]
        self.C = in_channels
        self.V = num_point
        self.AntiTCN = AntiTemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, C=self.C, V=self.V)
        self.fcn1 = nn.Linear(classifeature, in_channels * num_point * num_channels[0])

    def forward(self, x):
        x = self.fcn1(x)
        N, TCV = x.size()
        x = x.view(N, self.OT, self.C * self.V)
        deout = self.AntiTCN(x)
        return deout


class TCN_VAE(nn.Module):
    def __init__(self, en_input_size, en_num_channels, de_num_channels, output_size, num_person=1, in_channels=3, num_point=22,
                 fix_state=False, fix_weight=False):
        super(TCN_VAE, self).__init__()
        self.en_channels = en_num_channels
        self.de_channels = de_num_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.encoder = EncoderTCN(en_input_size, en_num_channels).to(device)
        self.decoder = DecoderATCN(output_size, de_num_channels, in_channels=in_channels, num_point=num_point).to(device)
        self.fix_state = fix_state
        self.fix_weight = fix_weight
        self.device = device
        if self.fix_weight:
            with torch.no_grad():
                # decoder fix weight
                self.decoder.TCN.requires_grad = False
                # self.decoder.out.requires_grad = False

        self.en_input_size = en_input_size

    def forward(self, x):
        x = x.cuda()
        mid = self.encoder(x)
        out = self.decoder(mid)

        return mid, out


if __name__ == "__main__":
    print('testing')

    # ### encoder debugging
    # x = torch.randn(10, 3, 100, 22, 1)
    # en = EncoderTCN(100, [50, 25, 10])
    # enout = en.forward(x)
    #
    # print(enout)
    # print(enout.shape)

    # ### decoder debugging
    # en = DecoderATCN(100, [10, 25, 50])
    # deout = en.forward(enout)
    #
    # print(deout)
    # print(deout.shape)

    ### TCN_VAE debugging
    x = torch.randn(10, 3, 100, 22, 1)
    tv = TCN_VAE(100, [75, 20, 10], [10, 20, 75], 100)
    mid, out = tv.forward(x)

    print(mid)
    print(mid.shape)
    # print('This is classfi:')
    # print(classfi)
    # print(classfi.shape)
    print(out)
    print(out.shape)
    #
    # modelData = "./tv.pth"  # 定义模型数据保存的路径
    #
    # torch.onnx.export(tv, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构

    # with SummaryWriter(log_dir='') as sw:  # 实例化 SummaryWriter ,可以自定义数据输出路径
    #     sw.add_graph(mstcn, (x,))  # 输出网络结构图
    #     sw.close()  # 关闭  sw

    # for name, param in mstcn.named_parameters():
    #     print(f'{name}: {param.numel()}')
    # print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))
