import netron
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from net.utils.Unsupervise_tools import TemporalConvNet, AntiTemporalConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out


class EncoderMSTCN(nn.Module):
    def __init__(self, C, num_output, num_person=1, in_channels=3, num_point=25, T=90, classifeature=128):
        super(EncoderMSTCN, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.mstcn = MultiScale_TemporalConv(C, num_output)
        self.fcn1 = nn.Linear(num_output * num_point * T, classifeature)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.mstcn(x)
        N, T, V, C = x.size()
        x = x.view(N, V * C * T)
        enout = self.fcn1(x)
        return enout


class DecoderATCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, in_channels=3, num_point=22, classifeature=128):
        super(DecoderATCN, self).__init__()
        self.in_channels = in_channels
        self.num_point = num_point
        self.OT = num_channels[0]
        self.CV = in_channels * num_point
        self.AntiTCN = AntiTemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, C=in_channels, V=num_point)
        self.fcn1 = nn.Linear(classifeature, in_channels * num_point * num_channels[0])

    def forward(self, x):
        x = self.fcn1(x)
        N, TCV = x.size()
        x = x.view(N, self.OT, self.CV)
        deout = self.AntiTCN(x)

        return deout


class MSTCN_VAE(nn.Module):
    def __init__(self, C, num_output, de_num_channels, output_size, num_person=1, in_channels=3,
                 num_point=22, fix_state=False, fix_weight=False):
        super(MSTCN_VAE, self).__init__()
        self.de_channels = de_num_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.encoder = EncoderMSTCN(C, num_output, num_person, in_channels, num_point).to(device)
        self.decoder = DecoderATCN(output_size, de_num_channels, in_channels=in_channels, num_point=num_point).to(device)
        self.fix_state = fix_state
        self.fix_weight = fix_weight
        self.device = device
        if self.fix_weight:
            with torch.no_grad():
                # decoder fix weight
                self.decoder.TCN.requires_grad = False
                # self.decoder.out.requires_grad = False

    def forward(self, x):
        x = x.cuda()
        mid = self.encoder(x)
        out = self.decoder(mid)

        return mid, out


if __name__ == "__main__":
    print('testing')

    ### encoder debugging
    # x = torch.randn(10, 3, 100, 22, 1)
    # en = EncoderMSTCN(3, 96)
    # enout = en.forward(x)
    #
    # print(enout.shape)

    # ### decoder debugging
    # x = torch.randn(10)
    # en = DecoderATCN(50, [15, 25, 30], 10)
    # deout = en.forward(x)
    #
    # print(deout)
    # print(deout.shape)

    ### TCN_VAE debugging
    x = torch.randn(1, 3, 100, 22, 1)

    mtv = MSTCN_VAE(3, 96, [1, 50], 100)
    mid, out = mtv.forward(x)

    print(mid)
    print(mid.shape)
    classfi = mid.view(mid.shape[0], 66)
    print('This is classfi:')
    print(classfi)
    print(classfi.shape)

    print(out)
    print(out.shape)

    # modelData = "./tv.pth"  # 定义模型数据保存的路径
    #
    # torch.onnx.export(mtv, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构

    # with SummaryWriter(log_dir='') as sw:  # 实例化 SummaryWriter ,可以自定义数据输出路径
    #     sw.add_graph(tv, (x,))  # 输出网络结构图
    #     sw.close()  # 关闭  sw

    # for name, param in mstcn.named_parameters():
    #     print(f'{name}: {param.numel()}')
    # print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))
