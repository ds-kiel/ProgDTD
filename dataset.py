# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from torchvision import transforms


class VimeoDataset(Dataset):
    def __init__(self, video_dir, text_split, transform):

        self.video_dir = video_dir
        self.text_split = text_split

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
            
        self.frames = []

        with open(self.text_split, 'r') as f:
            filenames = f.readlines()
            f.close()
            
        final_filenames = []
        for i in filenames:
            final_filenames.append(os.path.join(self.video_dir, i.split('\n')[0]))

        for f in final_filenames:
            try:
                frames = [os.path.join(f, i) for i in os.listdir(f)]
            except:
                continue
            # make sure images are in order, i.e. im1.png, im2.png, im3.png
            frames = sorted(frames)
            # make sure there are only 3 images in the Vimeo-90k triplet's folder for it to be a valid dataset sample
            self.frames.append(frames[0])
            self.frames.append(frames[1])
            self.frames.append(frames[2])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.frames[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
