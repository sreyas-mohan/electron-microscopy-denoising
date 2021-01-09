import os
import os.path
import numpy as np
import h5py
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import math

DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError(
                "Cannot register duplicate H5dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn


@register_dataset("bsd400")
def load_bsd400(data, batch_size=100, num_workers=0):
    data = os.path.join(data, "bsd400")
    train_dataset = H5Dataset(filename=os.path.join(data, "train.h5"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    valid_dataset = H5Dataset(filename=os.path.join(data, "valid.h5"))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, valid_loader, None





def get_transform(training, rotation_aug, resize_aug, image_size):
        
        
    def rotatedRectWithMaxArea(w, h, angle):
      """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle (maximal area) within the rotated rectangle.
      """
      if w <= 0 or h <= 0:
        return 0,0

      width_is_longer = w >= h
      side_long, side_short = (w,h) if width_is_longer else (h,w)

      # since the solutions for angle, -angle and 180-angle are all the same,
      # if suffices to look at the first quadrant and the absolute values of sin,cos:
      sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
      if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
      else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

      return wr,hr
    
    def randomrotate(x, angles):
        angle = angles[0] + (angles[1]-angles[0]) * np.random.random()
        xrot = x.rotate(angle, Image.NEAREST, expand = False, fillcolor= 0) 
        w, h = x.size
        wr, hr = rotatedRectWithMaxArea(w, h, math.radians(angle))
        return transforms.CenterCrop((hr, wr))(xrot)
        return xrot
    
    def randomresize(x, resize_min, resize_max):
        resize_factor = resize_min + (resize_max - resize_min)*np.random.random()
        w, h = x.size
        return transforms.functional.resize(x, 
                                            (int(w*resize_factor), int(h*resize_factor)), 
                                            Image.BILINEAR)
        
    transform = []
    transform.append(transforms.RandomHorizontalFlip()
                     ) if training else None
    transform.append(transforms.RandomVerticalFlip()
                     ) if training else None
    
    if resize_aug:
        transform.append(lambda x: randomresize(x, 0.75, 0.82)) if training else transform.append(lambda x: randomresize(x, 0.785, 0.785))

    if rotation_aug:
        transform.append(lambda x: randomrotate(x, angles=(-45, 45))) if training else transform.append(lambda x: randomrotate(x, angles=(28, 28)))

    transform.append(transforms.RandomCrop(
        image_size)) if training else None
    transform.append(transforms.ToTensor())
    return transforms.Compose(transform)



def load_ptceo2_v2(
        data,
        contrast='white',
        batch_size=100,
        image_size=400,
        num_workers=2,
        repeat_train=1, 
        test_batch_size=5, 
        rotation_aug = True,
        resize_aug = True, 
        generalization_exp = None, 
        allowed_gen_values = None):

    root_dir = data
    contrast = sorted(contrast.split('-'))
    contrast = '_'.join(contrast)

    if generalization_exp is not None:
        assert(generalization_exp in ['structure', 'defect'])
    if allowed_gen_values is not None:
        allowed_gen_values = sorted(allowed_gen_values.split('-'))

    train_dataset = ParticleDataset(
        os.path.join(
            root_dir,
            f'train_{contrast}.csv'),
        root_dir,
        get_transform(
            training=True, rotation_aug=rotation_aug, resize_aug=resize_aug, image_size=image_size), 
        generalization_exp = generalization_exp,
        allowed_gen_values = allowed_gen_values)
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset] * repeat_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)

    valid_dataset = ParticleDataset(
        os.path.join(
            root_dir,
            f'valid_{contrast}.csv'),
        root_dir,
        get_transform(
            training=False, rotation_aug=rotation_aug, resize_aug=resize_aug, image_size=image_size),
        generalization_exp = generalization_exp,
        allowed_gen_values = allowed_gen_values)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False)

    test_dataset = ParticleDataset(
        os.path.join(
            root_dir,
            f'test_{contrast}.csv'),
        root_dir,
        get_transform(
            training=False, rotation_aug=rotation_aug, resize_aug=resize_aug, image_size=image_size),
        generalization_exp = generalization_exp,
        allowed_gen_values = allowed_gen_values)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=1, shuffle=False)
    return train_loader, valid_loader, test_loader



@register_dataset("ptceo2")
def load_regular_dataloaders(
        data,
        contrast='white',
        batch_size=100,
        image_size=400,
        num_workers=2,
        repeat_train=1, 
        test_batch_size=5, 
        rotation_aug = True,
        resize_aug = True,
        generalization_exp = None, 
        allowed_gen_values = None):

    return load_ptceo2_v2(data = data, contrast = contrast, batch_size = batch_size, 
                            image_size = image_size, num_workers = num_workers, repeat_train = repeat_train,
                            test_batch_size = test_batch_size, rotation_aug = rotation_aug, resize_aug = resize_aug)

@register_dataset("ptceo2-structure")
def load_regular_dataloaders(
        data,
        contrast='white',
        batch_size=100,
        image_size=400,
        num_workers=2,
        repeat_train=1, 
        test_batch_size=5, 
        rotation_aug = True,
        resize_aug = True,
        generalization_exp = 'structure', 
        allowed_gen_values = None):

    return load_ptceo2_v2(data = data, contrast = contrast, batch_size = batch_size, 
                            image_size = image_size, num_workers = num_workers, repeat_train = repeat_train,
                            test_batch_size = test_batch_size, rotation_aug = rotation_aug, resize_aug = resize_aug,
                            generalization_exp = generalization_exp, allowed_gen_values = allowed_gen_values)

@register_dataset("ptceo2-defect")
def load_regular_dataloaders(
        data,
        contrast='white',
        batch_size=100,
        image_size=400,
        num_workers=2,
        repeat_train=1, 
        test_batch_size=5, 
        rotation_aug = True,
        resize_aug = True,
        generalization_exp = 'defect', 
        allowed_gen_values = None):

    return load_ptceo2_v2(data = data, contrast = contrast, batch_size = batch_size, 
                            image_size = image_size, num_workers = num_workers, repeat_train = repeat_train,
                            test_batch_size = test_batch_size, rotation_aug = rotation_aug, resize_aug = resize_aug,
                            generalization_exp = generalization_exp, allowed_gen_values = allowed_gen_values)




class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, generalization_exp=None, allowed_gen_values=None):
        super().__init__()
        self.metadata = pd.read_csv(csv_file, sep=",", header=0)
        
        if generalization_exp is not None:
            self.metadata[['structure', 'defect']] = self.metadata['particle'].str.split('-',expand=True)
            if generalization_exp == 'structure':
                self.metadata = self.metadata.loc[self.metadata['defect'].isin(['D0', 'D1'])]
            elif generalization_exp == 'defect':
                self.metadata = self.metadata.loc[self.metadata['structure'].isin(['PtNp1'])]
            self.metadata = self.metadata.loc[self.metadata[generalization_exp].isin(allowed_gen_values)]
            print(self.metadata.groupby(['structure', 'defect']).size())

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.metadata.iloc[index, 0])
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        particle, thickness, tilt_x, tilt_y, defocus, contrast = self.metadata.iloc[index, [
            1, 4, 5, 6, 10, 11]]
        return dict(
            image=image,
            name=f"{thickness}A-{tilt_x:02d}x-{tilt_y:02d}y-{defocus}nm",
            thickness=thickness,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            defocus=defocus,
            contrast=contrast,
            particle=particle)
