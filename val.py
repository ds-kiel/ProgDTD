import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torchvision import transforms
import seaborn as sns
import pandas as pd
import os
import torch.nn as nn
import torch.utils.data
import PIL
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM
from tqdm import tqdm
import glob
from ProgDTD import ScaleHyperpriorLightning, ScaleHyperprior
import yaml

device = 'cuda:0'



with open('params.yaml', "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

KODAK_dir = config['KODAK_dir']
    

def run_on_multiple_patches(x, model):
    patches_in = []
    patches_out = []
    patches_in.append(x[:, 0:256,256:512])
    patches_in.append(x[:, 0:256,0:256])
    patches_in.append(x[:, 256:512,0:256])
    patches_in.append(x[:, 256:512,256:512])

    x_fold = torch.zeros_like(x)
    model.to(device)

    for p in patches_in:
        p = p.to(device).view(-1, 3, 256, 256)
        x_hat, y_likelihoods, z_likelihoods = model(p)

        bpp_loss, distortion_loss, combined_loss = model.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, p
        )
        patches_out.append(x_hat.detach().clone())

    x_fold[:, 0:256,256:512] = patches_out[0]
    x_fold[:, 0:256,0:256] = patches_out[1]
    x_fold[:, 256:512,0:256] = patches_out[2]
    x_fold[:, 256:512,256:512] = patches_out[3]
    
    return x_fold, bpp_loss

def model_evalutation(model, dataset_path):
    
    MS_SSIM_LOSS = MS_SSIM(data_range=1, size_average=True, channel=3)
    SSIM_LOSS = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True) # channel=1 for grayscale images
    MSE_LOSS = nn.MSELoss()
    
    model.to(device)
    model.eval()
    
    bpp_loss_list = []
    PSNR_loss_list = []
    SSIM_loss_list = []
    MSSSIM_loss_list = []
    
    file_names = glob.glob(dataset_path)
    
    for im_path in file_names:
        
        x = PIL.Image.open(im_path).convert("RGB")
        x = transforms.Resize((512,512))(x)
        # x = transforms.CenterCrop((512,512))(x)
        x = transforms.ToTensor()(x)
        x = x.view(3, 512, 512)
        
        x_hat, bpp_loss = run_on_multiple_patches(x, model)
        
        x = x.view(-1, 3, 512, 512)
        x_hat = x_hat.view(-1, 3, 512, 512)

        ####
        psnr = 20 * math.log10(1 / np.sqrt(MSE_LOSS(x_hat, x).item()))
        PSNR_loss_list.append(psnr)

        msssim = -10 * np.log10(1 - MS_SSIM_LOSS(x_hat, x).item())
        MSSSIM_loss_list.append(msssim)
        
        ssim = -10 * np.log10(1 - SSIM_LOSS(x_hat, x).item())
        SSIM_loss_list.append(ssim)
        ###
        
        bpp_loss_list.append(bpp_loss.item())

    return [np.mean(bpp_loss_list),
            np.mean(PSNR_loss_list),
            np.mean(MSSSIM_loss_list),
            np.mean(SSIM_loss_list)
           ]

###

def ProgDTD(model_path, p, Lambda, dataset_path):
    model = ScaleHyperpriorLightning(
        model=ScaleHyperprior(network_channels=128, compression_channels=192),
        distortion_lambda=Lambda,
    )
    model = torch.load(model_path) 
    
    model.model.p_latent = p
    model.model.p_hyper_latent = model.model.p_latent
    
    images_size, psnr, msssim, ssim = model_evalutation(model, dataset_path)
    torch.cuda.empty_cache()
    del model
    
    return images_size, psnr, msssim, ssim


def Evaluation(Lambda, metrics, prog_range, dataset_path):
    res = []
    for i in tqdm([1, 5,10, 15, 20, 25, 30, 40, 50 , 60, 70, 80, 85, 90, 95, 100]):
        model_path = f'Lambda={Lambda} - range={prog_range}'
        print(model_path)
        images_size, psnr, msssim, ssim = ProgDTD(model_path, i/100, Lambda, dataset_path)
        metrics['prog_range'].append(prog_range)
        metrics['Lambda'].append(Lambda)
        metrics['bpp'].append(images_size)
        metrics['psnr'].append(psnr)
        metrics['msssim'].append(msssim)
        metrics['ssim'].append(ssim)
    return metrics



def main():
    
    sns.set(rc={'figure.figsize':(6,4.5),
                'pdf.fonttype':42, 'ps.fonttype':42
               })
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set_style("whitegrid")
    plt.rc('legend', fontsize=12)


    metrics={
        'prog_range':[],
        'Lambda':[],
        'bpp':[],
        'psnr':[],
        'msssim':[],
        'ssim':[],
    }

    dataset_path = KODAK_dir
    Evaluation(Lambda=0.01, metrics=metrics, prog_range='[0.0-1.0]', dataset_path=dataset_path)
    Evaluation(Lambda=0.1, metrics=metrics, prog_range='[0.0-1.0]', dataset_path=dataset_path)
    Evaluation(Lambda=1.0, metrics=metrics, prog_range='[0.0-1.0]', dataset_path=dataset_path)
    Evaluation(Lambda=0.01, metrics=metrics, prog_range='[0.3-1.0]', dataset_path=dataset_path)
    Evaluation(Lambda=0.1, metrics=metrics, prog_range='[0.3-1.0]', dataset_path=dataset_path)
    Evaluation(Lambda=1.0, metrics=metrics, prog_range='[0.3-1.0]', dataset_path=dataset_path)


    # MS-SSIM
    df = pd.DataFrame.from_dict(metrics)
    sns.lineplot(data=df.query('Lambda==0.01 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 0.01 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.1 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 0.1 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==1.0 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 1.0 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.01 and prog_range=="[0.3-1.0]" and msssim > 12'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 0.01 U(0.3,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.1 and prog_range=="[0.3-1.0]" and msssim > 15'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 0.1 U(0.3,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==1.0 and prog_range=="[0.3-1.0]" and msssim > 15'),
                 x="bpp", y="msssim", marker='o', label='ProgDTD 1.0 U(0.3,1)', linewidth=1)

    plt.ylabel(ylabel='MS-SSIM (dB scale)')
    plt.savefig('MS-SSIM.pdf', bbox_inches = 'tight')
    plt.close()

    # PSNR
    sns.lineplot(data=df.query('Lambda==0.01 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 0.01 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.1 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 0.1 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==1.0 and prog_range=="[0.0-1.0]"'),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 1.0 U(0,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.01 and prog_range=="[0.3-1.0]" and psnr > 26 '),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 0.01 U(0.3,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==0.1 and prog_range=="[0.3-1.0]" and psnr > 30 '),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 0.1 U(0.3,1)', linewidth=1)

    sns.lineplot(data=df.query('Lambda==1.0 and prog_range=="[0.3-1.0]" and psnr > 30 '),
                 x="bpp", y="psnr", marker='o', label='ProgDTD 1.0 U(0.3,1)', linewidth=1)

    plt.ylabel(ylabel='PSNR (dB scale)')
    plt.savefig('PSNR.pdf', bbox_inches = 'tight')
    plt.close()



if __name__ == "__main__":
    main()