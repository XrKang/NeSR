import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import utils
import random
from AWAN_INR_AT_coord import *
import torch.nn.functional as F

from scipy.io import savemat

import os
from functools import partial
import pickle
from skimage import measure, color

parser = argparse.ArgumentParser(description="HSI Rec Eval")
parser.add_argument("--mat_path_valid", type=str,
                    default='./test_mat61',
                    help="HyperSet path")

parser.add_argument("--rgb_path_valid", type=str,
                    default='./rgb',
                    help="RGBSet path")

parser.add_argument("--model_path", type=str,
                    default="../pth/model.pth",
                    help="model path")

parser.add_argument('--save_dir', type=str,
                    default='./test_result')

parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()

def load_parallel(model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = measure.compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

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


def valid(arg, model):
    torch.cuda.empty_cache()
    val_set = utils.HyperValid_ICVL(arg)
    val_set_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)
    MRAE_epoch = 0
    RMSE_epoch = 0
    MRAE_epoch_noPatch = 0
    RMSE_epoch_noPatch = 0
    model.eval()

    H = (1392//4)*4
    W = (1300//4)*4

    h_crop = 128
    w_crop = 128

    save_path = arg.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for iteration, (rgb_list, hsi_list) in enumerate(val_set_loader):
        MRAEs = []
        RMSEs = []

        results = []
        gts = []

        for index in range(len(rgb_list)):

            rgb = rgb_list[index]
            hsi = hsi_list[index]

            sample_step = 10  # nm
            bands_num = int(300 // sample_step + 1)
            B, ch, H_p, W_p = hsi.shape
            hsi = torch.unsqueeze(hsi, dim=0)
            hsi_down = F.interpolate(hsi, size=(bands_num,  H_p, W_p), mode='trilinear')
            hsi_down = torch.squeeze(hsi_down, dim=0)
            _, bands_num, _, _ = hsi_down.shape
            coor = make_coord([bands_num, H_p, W_p]). \
                permute(3, 0, 1, 2).unsqueeze(0).expand(B, 3, *[bands_num, H_p, W_p]).cuda()

            if arg.cuda:
                rgb = rgb.cuda()
                hsi_down = hsi_down.cuda()
            with torch.no_grad():
                pred = model(rgb, coor)
                RMAE_patch = computeMRAE(pred, hsi_down)
                MRAEs.append(RMAE_patch)

                RMSE_patch = computeRMSE(pred, hsi_down)
                RMSEs.append(RMSE_patch)

                results.append(pred)
                gts.append(hsi_down)


        MRAE_one = np.mean(MRAEs)
        RMSE_one = np.mean(RMSEs)
        MRAE_epoch = MRAE_epoch + MRAE_one
        RMSE_epoch = RMSE_epoch + RMSE_one
        print("VAL===> Val.MRAE: {:.8f} RMSE: {:.8f} ".format(MRAE_one, RMSE_one))

        result = np.zeros([bands_num, H, W]).astype(np.float64)
        gt = np.zeros([bands_num, H, W]).astype(np.float64)
        idx_save = 0
        for idx_H in range(0, H, h_crop):
            for idx_W in range(0, W, w_crop):
                recovered = results[idx_save][0].cpu()
                groundTruth = gts[idx_save][0].cpu()
                recovered = recovered.detach().numpy()
                groundTruth = groundTruth.detach().numpy()

                gt[:, idx_H: idx_H + h_crop, idx_W: idx_W + w_crop] = groundTruth
                result[:, idx_H: idx_H + h_crop, idx_W: idx_W + w_crop] = recovered

                idx_save +=1

        MRAE_noPatch = computeMRAE_all(result, gt)
        RMSE_noPatch = computeRMSE_all(result, gt)

        MRAE_epoch_noPatch = MRAE_epoch_noPatch + MRAE_noPatch
        RMSE_epoch_noPatch = RMSE_epoch_noPatch + RMSE_noPatch
        print("VAL_noPatch===> Val.MRAE: {:.8f} RMSE: {:.8f} ".format(MRAE_noPatch, RMSE_noPatch))


        gt_dic = {'HyperImage': gt}
        gt_name = os.path.join(save_path, str(iteration)+'_gt.mat')
        savemat(gt_name, gt_dic, do_compression=True)
        savemat(gt_name, gt_dic)

        pred_dic = {'HyperImage': result}
        pred_name = os.path.join(save_path, str(iteration)+'_our.mat')
        savemat(pred_name, pred_dic, do_compression=True)
        savemat(pred_name, pred_dic)

    MRAE_valid = MRAE_epoch / (iteration + 1)
    RMSE_valid = RMSE_epoch / (iteration + 1)

    MRAE_valid_noPatch = MRAE_epoch_noPatch / (iteration + 1)
    RMSE_valid_noPatch = RMSE_epoch_noPatch / (iteration + 1)

    print("VAL===> Val_Avg. MRAE: {:.8f} RMSE: {:.8f}".format(MRAE_valid, RMSE_valid))
    print("VAL_noPatch===> Val_Avg. MRAE: {:.8f} RMSE: {:.8f}".format(MRAE_valid_noPatch, RMSE_valid_noPatch))


def computeMRAE_all(recovered, groundTruth):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = np.abs(groundTruth - recovered) / (groundTruth+0.0001)
    mrae = np.mean(difference)

    return mrae


def computeRMSE_all(recovered, groundTruth):
    """
    Compute RMSE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: RMSE between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = (groundTruth - recovered) ** 2
    rmse = np.sqrt(np.mean(difference))

    return rmse




def computeMRAE(recovered, groundTruth):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """
    recovered = recovered.cpu()
    groundTruth = groundTruth.cpu()
    recovered = recovered.detach().numpy()
    groundTruth = groundTruth.detach().numpy()

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = np.abs(groundTruth - recovered) / (groundTruth+0.0001)
    mrae = np.mean(difference)

    return mrae


def computeRMSE(recovered, groundTruth):
    """
    Compute RMSE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: RMSE between `recovered` and `groundTruth`.
    """
    recovered = recovered.cpu()
    groundTruth = groundTruth.cpu()
    recovered = recovered.detach().numpy()
    groundTruth = groundTruth.detach().numpy()

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = (groundTruth - recovered) ** 2
    rmse = np.sqrt(np.mean(difference))

    return rmse


def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()

    p = [measure.compare_psnr(img1_np[k, :, :], img2_np[k, :, :]) for k in range(img2.shape[0])]
    mean_p = sum(p) / (len(p) + 1)
    return mean_p



def main(opt):
    torch.cuda.empty_cache()
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = opt.model_path



    model = ImplicitRepresentation(opt)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        model.load_state_dict(load_parallel(model_path))
    model.eval()

    if opt.cuda:
        model.cuda()
    with torch.no_grad():
        valid(opt, model)


if __name__ == '__main__':
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    print(opt.model_path)
    opt.in_ch = 3
    opt.embed_ch = 256
    opt.n_DRBs = 8

    opt.hidden_list = [32, 16, 128, 128, 256, 256]

    opt.imnet_in_dim = 32+3
    opt.imnet_out_dim = 1
    opt.numb_MultiHead = 2

    print(opt)
    main(opt)

