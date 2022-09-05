from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from numpy import nan

from .converter import Converter
from .dataset import CarlaDataset
from .dataset_wrapper import Wrap
from . import common

class MaskDataset(CarlaDataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), kernel=11, num_select=10, full_replace=False):
        super().__init__(dataset_dir, transform=transform)
        self.kernel = kernel
        self.blur = transforms.GaussianBlur(kernel)
        self.num_select = num_select
        self.full_replace = full_replace

    def blur_random(self, mask, image_tensor, blurred_image, inside=False):
        # removes the RGB channels
        mask = mask.reshape(3,3,image_tensor.shape[1], image_tensor.shape[2])
        sum_mask = mask.sum(dim=1)
        sum_mask[sum_mask > 1] = 1
        # find the locations which are valid to blur by applying blur as a filter
        if inside:
            usable = sum_mask # pick a location in the masked area
        else: # pick a location outside the masked area with no overlap
            usable = self.blur(sum_mask.float())
            usable[usable > 0] = 1
            # same inversion code to invert the locations
            usable = (usable.int() + 1)
            usable[usable == 2] = 0

        # get some random valid locations to apply the blur
        indices = torch.randperm(len(usable.nonzero()))

        # apply blur at selected locations
        usable = usable.nonzero()[indices[:self.num_select]].numpy()

        # create a circular mask at the selected locations
        sum_mask = np.zeros(mask.shape) # sum mask redefined here
        sum_mask = np.sum(sum_mask, axis=1)
        for center in usable:
            y,x = np.ogrid[-center[1]:sum_mask.shape[1]-center[1], -center[2]:sum_mask.shape[2]-center[2]]
            keep = x*x + y*y <= 100
            sum_mask[center[0], keep] = 1 # select a circle of pixels
        sum_mask[sum_mask > 1] = 1
        sum_mask = torch.tensor(sum_mask)
        sum_mask = torch.stack([sum_mask.clone(), sum_mask.clone(), sum_mask.clone()], dim=1) # expand to RGB channels
        # print(sum_mask.shape)
        sum_mask = sum_mask.reshape(9,image_tensor.shape[1], image_tensor.shape[2])

        inverted_mask = (sum_mask.int() + 1)
        inverted_mask[inverted_mask == 2] = 0
        inverted_mask= inverted_mask.bool()

        return blurred_image * sum_mask + image_tensor * inverted_mask

    def __getitem__(self, i):
        image_tensor, topdown, points, target, actions, meta = super().__getitem__(i)
        
        path = self.dataset_dir
        frame = self.frames[i]
        meta = '%s %s' % (path.stem, frame)

        # mask = Image.open(path / 'masks' / ('%s.png' % frame))
        # create a fake mask (remove this when real masks are implemented)
        mask = np.zeros(tuple(image_tensor.size()))
        mask = mask.reshape(3,3,image_tensor.shape[1],image_tensor.shape[2])
        mask = np.sum(mask, axis=1) # removes the axis corresponding to color
        usable = np.array([[np.random.randint(3), np.random.randint(image_tensor.shape[1]), np.random.randint(image_tensor.shape[2])] for i in range(2)])
        usable = np.array([[0, 48, 158], [0, 100, 52]])
        for center in usable:
            y,x = np.ogrid[-center[1]:mask.shape[1]-center[1], -center[2]:mask.shape[2]-center[2]]
            keep = x*x + y*y <= 300
            mask[center[0], keep] = 1 # select a circle of pixels
        # mask[mask < 0.045] = 0
        mask = torch.tensor(mask)
        # FAKE MASK CODE COMPLETE

        # transform real mask to tensor
        # mask = transforms.functional.to_tensor(mask)

        # combine masks
        mask = torch.stack([mask.clone(), mask.clone(), mask.clone()], dim=1) # expand to RGB channels
        mask = mask.reshape(9,image_tensor.shape[1], image_tensor.shape[2])

        # create inverted mask
        inverted_mask = (mask.int() + 1)
        inverted_mask[inverted_mask == 2] = 0
        inverted_mask= inverted_mask.bool()

        blurred_image = self.blur(image_tensor)
        # blurred_image = torch.ones(image_tensor.shape) # used for debugging 

        # masking mode 1: full replace
        if self.full_replace:
            positive_image = image_tensor * mask + blurred_image * inverted_mask
            negative_image = blurred_image * mask + image_tensor * inverted_mask


        # masking mode 2: partial replace
        # create reachable locations
        else:
            positive_image = self.blur_random(mask, image_tensor, blurred_image, inside=False) 
            negative_image = self.blur_random(mask, image_tensor, blurred_image, inside=True) 
        return (image_tensor, positive_image, negative_image), topdown, points, target, actions, meta

if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw
    from .utils.heatmap import ToHeatmap

    # for path in sorted(Path('/home/bradyzhou/data/carla/carla_challenge_curated').glob('*')):
        # data = CarlaDataset(path)

        # for i in range(len(data)):
            # data[i]

    data = MaskDataset(sys.argv[1], kernel=11, num_select=3)
    converter = Converter()
    to_heatmap = ToHeatmap()

    for i in range(len(data)):
        rgb, topdown, points, target, actions, meta = data[i]
        rgb = rgb[1] # only showing the "positive" sample
        points_unnormalized = (points + 1) / 2 * 256
        points_cam = converter(points_unnormalized)

        target_cam = converter(target)

        heatmap = to_heatmap(target[None], topdown[None]).squeeze()
        heatmap_cam = to_heatmap(target_cam[None], rgb[None]).squeeze()

        _heatmap = heatmap.cpu().squeeze().numpy() / 10.0 + 0.9
        _heatmap_cam = heatmap_cam.cpu().squeeze().numpy() / 10.0 + 0.9

        _rgb = (rgb.cpu() * 255).byte().numpy().transpose(1, 2, 0)[:, :, :3]
        _rgb[heatmap_cam > 0.1] = 255
        _rgb = Image.fromarray(_rgb)

        _topdown = common.COLOR[topdown.argmax(0).cpu().numpy()]
        _topdown[heatmap > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw_map = ImageDraw.Draw(_topdown)
        _draw_rgb = ImageDraw.Draw(_rgb)

        for x, y in points_unnormalized:
            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in converter.cam_to_map(points_cam):
            _draw_map.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        for x, y in points_cam:
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)
        _rgb = cv2.resize(np.hstack((_rgb, _topdown)), (_rgb.size[0] * 4, _rgb.size[1] * 4), fx = 0.1, fy = 0.1
                         ,interpolation = cv2.INTER_CUBIC)
        # cv2.imshow('debug', cv2.cvtColor(_rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('debug', cv2.cvtColor(_rgb, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1000)