# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision import transforms
import pytorch_lightning as pl
import random
import os
import yaml

from dataset import VimeoDataset
from ProgDTD import ScaleHyperpriorLightning, ScaleHyperprior


device = 'cuda:0'
print(torch.cuda.is_available())


torch.manual_seed(0)
random.seed(10)
np.random.seed(0)

with open('params.yaml', "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

batch_size = config['batch_size']
crop = config['crop']
resize = config['resize']
Lambda = config['Lambda']
vimeo_train_dir = config['vimeo_train_dir']
epoch = config['epoch']
save_dir = config['save_dir']
progressiveness_range=config['save_dir']

#--**--#

train_transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.Resize((resize,resize)),
    transforms.ToTensor(),
])

seq_dir = os.path.join(vimeo_train_dir, 'sequences')
train_txt = os.path.join(vimeo_train_dir, 'tri_trainlist.txt')
VIMEO_data = VimeoDataset(video_dir=seq_dir, text_split=train_txt, transform=train_transform)

VIMEO_data, _ = torch.utils.data.random_split(VIMEO_data, [150_000, 153939 - 150_000], generator=torch.Generator().manual_seed(41))

VIMEO_training_set, VIMEO_remaining = torch.utils.data.random_split(VIMEO_data, [140_000, 10_000], generator=torch.Generator().manual_seed(41))
VIMEO_validation_set, VIMEO_test_set = torch.utils.data.random_split(VIMEO_remaining, [5_000, 5_000], generator=torch.Generator().manual_seed(41))

VIMEO_training_set.dataset.transform = train_transform
VIMEO_validation_set.dataset.transform = test_transform
VIMEO_test_set.dataset.transform = test_transform


#--**--#

model = ScaleHyperpriorLightning(
    model=ScaleHyperprior(network_channels=128, compression_channels=192, progressiveness_range=[0.3, 1.0]),
    distortion_lambda=Lambda,
)
model.to(device)



#--**--#

train_loader = torch.utils.data.DataLoader(VIMEO_training_set, batch_size=batch_size, num_workers=8)
val_loader = torch.utils.data.DataLoader(VIMEO_validation_set, batch_size=batch_size, num_workers=8)
test_loader = torch.utils.data.DataLoader(VIMEO_test_set, batch_size=batch_size, num_workers=8)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs = epoch,
    enable_checkpointing=False
                    )
trainer.fit(model, train_loader, val_loader)

#--**--#
torch.save(model, save_dir) 
