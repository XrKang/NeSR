import torch
from torch import nn
from torch.nn import functional as F
import math


class AWCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=8, dimension=2, sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // reduction

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        f = self.count_cov_second(theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y


class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        # nonlocal module
        self.non_local = NONLocalBlock2D(channels)

    def forward(self, x):
        # divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class DRAB(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, k1_size=3, k2_size=1, dilation=1):
        super(DRAB, self).__init__()
        self.conv1 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu2 = nn.PReLU()
        # T^{l}_{1}: (conv.)
        self.up_conv = Conv3x3(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.up_relu = nn.PReLU()
        self.se = AWCA(res_dim)
        # T^{l}_{2}: (conv.)
        self.down_conv = Conv3x3(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.down_relu = nn.PReLU()

    def forward(self, x, res):
        x_r = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu2(x)
        # T^{l}_{1}
        x = self.up_conv(x)
        x += res
        x = self.up_relu(x)
        res = x
        x = self.se(x)
        # T^{l}_{2}
        x = self.down_conv(x)
        x += x_r
        x = self.down_relu(x)
        return x, res


class AWAN(nn.Module):
    def __init__(self, inplanes=3, channels=256, n_DRBs=8):
        super(AWAN, self).__init__()
        # 2D Nets
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.PReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)

        self.backbone = nn.ModuleList(
            [DRAB(in_dim=channels, out_dim=channels, res_dim=channels, k1_size=5, k2_size=3, dilation=1) for _ in
             range(n_DRBs)])

        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, channels, 3, 1)
        self.tail_nonlocal = PSNL(channels)

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        residual = out
        res = out

        for i, block in enumerate(self.backbone):
            out, res = block(out, res)

        out = self.tail_conv2D(out)
        out = torch.add(out, residual)
        out = self.output_conv2D(self.output_prelu2D(out))
        out = self.tail_nonlocal(out)
        return out



class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value):
        scores = torch.matmul(query.permute(1, 0), key)
        p_attn = F.softmax(scores)
        value = value.contiguous().permute(1, 0)
        p_val = torch.matmul(p_attn, value)
        p_val = p_val.contiguous().permute(1, 0)
        return p_val

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        hidden_list = args.hidden_list
        self.numb_MultiHead = args.numb_MultiHead

        layers_q = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_q.append(nn.Linear(lastv, hidden))
            layers_q.append(nn.ReLU())
            lastv = hidden

        layers_q.append(nn.Linear(lastv, lastv))
        self.layers_q = nn.Sequential(*layers_q)

        layers_k = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_k.append(nn.Linear(lastv, hidden))
            layers_k.append(nn.ReLU())
            lastv = hidden

        layers_k.append(nn.Linear(lastv, lastv))
        self.layers_k = nn.Sequential(*layers_k)

        layers_v = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_v.append(nn.Linear(lastv, hidden))
            layers_v.append(nn.ReLU())
            lastv = hidden

        layers_v.append(nn.Linear(lastv, lastv))
        self.layers_v = nn.Sequential(*layers_v)

        self.attention = Attention()

        layers2 = []
        lastv = lastv * self.numb_MultiHead
        for hidden in hidden_list[2:]:
            layers2.append(nn.Linear(lastv, hidden))
            layers2.append(nn.ReLU())
            lastv = hidden
        layers2.append(nn.Linear(lastv, out_dim))
        self.layers2 = nn.Sequential(*layers2)

    def forward(self, x):
        shape = x.shape[:-1]
        query = self.layers_q(x.view(-1, x.shape[-1]))
        key = self.layers_k(x.view(-1, x.shape[-1]))
        value = self.layers_v(x.view(-1, x.shape[-1]))

        x_att = []
        for idx in range(self.numb_MultiHead):
            x = self.attention(query, key, value)
            x_att.append(x)
        output = torch.cat(x_att, 1)

        output = self.layers2(output)

        return output.view(*shape, -1)


class ImplicitRepresentation(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.inplanes = args.in_ch
        self.channels = args.embed_ch
        self.n_DRBs = args.n_DRBs
        self.feature_encoder = AWAN(self.inplanes, self.channels, self.n_DRBs)

        self.conv_3d_1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv_3d_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.imnet_in_dim = args.imnet_in_dim
        self.imnet_out_dim = args.imnet_out_dim
        self.imnet = MLP(self.imnet_in_dim, self.imnet_out_dim, args)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)



    def forward(self, x, coor):
        # 获取LR的feature和coordinate
        feat = self.feature_encoder(x)  # bs,nf, h, w
        bs, _, bands, H, W = coor.shape

        _, C0, _, _ = feat.shape

        feat_HC_pre = feat.permute(0, 3, 2, 1)  # (b,w,h,ch)
        feat_WC_pre = feat.permute(0, 2, 3, 1)  # (b,h,w,ch)

        feat_HC = F.upsample(feat_HC_pre, [H, bands], mode="nearest")
        feat_WC = F.upsample(feat_WC_pre, [W, bands], mode="nearest")

        feat_HC = feat_HC.permute(0, 3, 2, 1)   # (b,bands,h,w)
        feat_HC = feat_HC.unsqueeze(dim=1)      # (b,1, bands,h,w)
        # print("feat_HC", feat_HC.shape)
        feat_WC = feat_WC.permute(0, 3, 1, 2)   # (b,bands,h,w)
        feat_WC = feat_WC.unsqueeze(dim=1)      # (b,1,bands,h,w)
        # print("feat_WC", feat_WC.shape)

        feat_inp = torch.cat([feat_HC, feat_WC], dim=1) # (b,2,bands,h,w)
        # print("feat_inp", feat_inp.shape)
        feat_inp_3d = self.lrelu(self.conv_3d_2(self.conv_3d_1(feat_inp)))
        feat_inp_3d = torch.cat([feat_inp_3d, coor], dim=1)
        # feat_inp_3d = self.lrelu((self.conv_3d_1(feat_inp)))
        # print("feat_inp_3d", feat_inp_3d.shape)     # (b,16,bands,h,w)

        feat_inp_3d = feat_inp_3d.permute(0, 2, 3, 4, 1)    # (b,bands,h,w,16)
        bs, bands, H, W, _ = feat_inp_3d.shape
        # print("feat_inp_3d", feat_inp_3d.shape)
        feat_mlp_inp= feat_inp_3d.contiguous().view(bs * bands * H * W, -1)
        pred = self.imnet(feat_mlp_inp).contiguous().view(bs, bands, H, W, -1)

        pred = pred[:, :, :, :, 0]  # (b,bands,h,w)
        # print("pred", pred.shape)

        return pred

def make_coord(shape, ranges=None,):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()    # size->(H,) range([-1, 1) center(=0.00
        # r=1/H
        # seq=(((2/H * arr[0:H-1])->arr[0:2*(H-1)/H] + (-1))->arr[-1:(2*H-2)/H)-1]) + (1/H)) -> arr[-1/H:1/H].size(H,)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # ret->size[H,W] range([-1/H, 1/H], [-1/W, 1/W]), center(1/H,1/W)=0.0.0
    return ret
    # If not flatten: (H,W,2)

if __name__ == "__main__":
    from argparse import Namespace

    from thop.profile import profile


    args = Namespace()

    args.hidden_list = [32, 16, 128, 128, 256, 256]
    args.numb_MultiHead = 2

    args.in_ch = 3
    args.embed_ch = 64
    args.n_DRBs = 8

    args.imnet_in_dim = 32+3
    args.imnet_out_dim = 1

    input_tensor = torch.rand(2, 3, 64, 32).cuda()
    coord = make_coord([31, 64, 32]).cuda()
    coord = coord.permute(3, 0, 1, 2).unsqueeze(0).expand(2, 3, *[31, 64, 32, ]).cuda()
    # print(coord.shape)

    # model = AWAN(3, 31, 200, 10)
    model = ImplicitRepresentation(args)
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor, coord)
    print(output_tensor.size())
    # name = "aaa"
    # total_ops, total_params = profile(model, (input_tensor,))
    # print("%s         | %.2f(M)      | %.2f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    # print('Parameters number is %.2f(M) '% (sum(param.numel() for param in model.parameters())/ (1000 ** 2)))
    # print(torch.__version__)
