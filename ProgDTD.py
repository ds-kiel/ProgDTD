import numpy as np
import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data
from typing import Dict, List, Optional, Sequence, Tuple
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch.optim as optim
import torch.nn.functional as F


from blocks import ImageAnalysis, HyperAnalysis, HyperSynthesis, ImageSynthesis


class ScaleHyperprior(nn.Module):
    def __init__(
        self,
        network_channels: Optional[int] = None,
        compression_channels: Optional[int] = None,
        image_analysis: Optional[nn.Module] = None,
        image_synthesis: Optional[nn.Module] = None,
        image_bottleneck: Optional[nn.Module] = None,
        hyper_analysis: Optional[nn.Module] = None,
        hyper_synthesis: Optional[nn.Module] = None,
        hyper_bottleneck: Optional[nn.Module] = None,
        progressiveness_range: Optional[List] = None,
    ):
        super().__init__()
        self.image_analysis = ImageAnalysis(network_channels, compression_channels)  
        self.hyper_analysis = HyperAnalysis(network_channels, compression_channels) 
        self.hyper_synthesis = HyperSynthesis(network_channels, compression_channels)  
        self.image_synthesis = ImageSynthesis(network_channels, compression_channels)
        
        self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)
        self.image_bottleneck = GaussianConditional(scale_table=None)
        self.progressiveness_range = progressiveness_range
        self.p_hyper_latent = None
        self.p_latent = None
        
    def forward(self, images):
            
        self.latent = self.image_analysis(images)
        self.hyper_latent = self.hyper_analysis(self.latent)
        
        #---***---#
        self.latent = self.rate_less_latent(self.latent)
        self.hyper_latent = self.rate_less_hyper_latent(self.hyper_latent)
        #---***---#

        
        self.noisy_hyper_latent, self.hyper_latent_likelihoods = self.hyper_bottleneck(
            self.hyper_latent
        )

        self.scales = self.hyper_synthesis(self.noisy_hyper_latent)
        self.noisy_latent, self.latent_likelihoods = self.image_bottleneck(self.latent, self.scales)
        
        #---***---#
        self.latent_likelihoods = self.drop_zeros_likelihood(self.latent_likelihoods, self.latent)
        self.hyper_latent_likelihoods = self.drop_zeros_likelihood(self.hyper_latent_likelihoods, self.hyper_latent)
        #---***---#
        
        self.reconstruction = self.image_synthesis(self.noisy_latent)

        self.rec_image = self.reconstruction.detach().clone()

        return self.reconstruction, self.latent_likelihoods, self.hyper_latent_likelihoods



    def rate_less_latent(self, data):
        self.save_p = []
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_latent:
                # p shows the percentage of keeping
                p = self.p_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1],1)[0]
                self.save_p.append(p)

            if p == 1.0:
                pass            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data
    
    def rate_less_hyper_latent(self, data):
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_hyper_latent:
                # p shows the percentage of keeping
                p = self.p_hyper_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
                p = self.save_p[i]
            if p == 1.0:
                pass
            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data

    def drop_zeros_likelihood(self, likelihood, replace):
        temp_data = likelihood.clone()
        temp_data = torch.where(
            replace == 0.0,
            torch.cuda.FloatTensor([1.0])[0],
            likelihood,
        )
        return temp_data
    
    
    
    
class ScaleHyperpriorLightning(pl.LightningModule):
    def __init__(
        self,
        model: ScaleHyperprior,
        distortion_lambda,
    ):
        super().__init__()

        self.model = model
        self.distortion_lambda = distortion_lambda


    def forward(self, images):
        return self.model(images)
        
    def training_step(self, batch, batch_idx):
        
        images = batch

        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "train_loss": combined_loss.item(),
                "train_distortion_loss": distortion_loss.item(),
                "train_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        return {
            "loss": combined_loss,
           }


    def training_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('train_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

        # normal_imshow(self.model.rec_image[0].to('cpu').detach().numpy())
        # plt.show()

    def validation_step(self, batch, batch_idx):
        
        self.model.p_hyper_latent = .2
        self.model.p_latent = .2
        
        images = batch
        
        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "val_loss": combined_loss.item(),
                "val_distortion_loss": distortion_loss.item(),
                "val_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        self.model.p_hyper_latent = None
        self.model.p_latent = None

        return {
            "loss": combined_loss,
           }


    def validation_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('val_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

        return {
                "optimizer": optimizer,
            }

        
    def rate_distortion_loss(self, reconstruction, latent_likelihoods,
                             hyper_latent_likelihoods, original,):
        
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width

        bits = (
            latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()
        ) / -math.log(2)
        
        bpp_loss = bits / num_pixels

        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss

        return bpp_loss, distortion_loss, combined_loss

