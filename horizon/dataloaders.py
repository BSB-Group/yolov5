"""
SEA.AI
authors: Kevin Serrano,

This file contains the dataset classes used for training 
YOLOv5 horizon detection model (RGB and IR16bit).
"""

import numpy as np
import fiftyone as fo
import cv2

from torch.utils.data import Dataset, DataLoader
import horizon.transforms as T


def get_train_rgb_dataloader(dataset: fo.Dataset,
                             imgsz: int,
                             batch_size: int = 16,
                             num_workers: int = 4,
                             shuffle: bool = True,
                             pin_memory: bool = False,
                             dataloader_kwargs: dict = {},
                             ):
    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_augment_RGB(imgsz),
        target_transform=T.points_to_normalised_pitch_theta(imgsz),
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **dataloader_kwargs,
                      )


def get_val_rgb_dataloader(dataset: fo.Dataset,
                           imgsz: int,
                           batch_size: int = 16,
                           num_workers: int = 4,
                           shuffle: bool = False,
                           pin_memory: bool = False,
                           dataloader_kwargs: dict = {},
                           ):
    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_base_RGB(imgsz),
        target_transform=T.points_to_normalised_pitch_theta(imgsz),
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **dataloader_kwargs,
                      )


def get_train_IR16bit_dataseloader(dataset: fo.Dataset,
                               imgsz: int,
                               batch_size: int = 64,
                               num_workers: int = 4,
                               shuffle: bool = True,
                               pin_memory: bool = False,
                               dataloader_kwargs: dict = {},
                               ):
    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_augment_IR16bit(imgsz),
        target_transform=T.points_to_normalised_pitch_theta(imgsz),
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **dataloader_kwargs,
                      )


def get_val_IR16bit_dataloader(dataset: fo.Dataset,
                           imgsz: int,
                           batch_size: int = 64,
                           num_workers: int = 4,
                           shuffle: bool = False,
                           pin_memory: bool = False,
                           dataloader_kwargs: dict = {},
                           ):
    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_base_IR16bit(imgsz),
        target_transform=T.points_to_normalised_pitch_theta(imgsz),
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **dataloader_kwargs,
                      )


class FiftyOneBaseDataset(Dataset):
    """
    Base class for datasets using FiftyOne datasets.
    Use this as parent class for your custom dataset.

    You may want to extend __init__ and __getitem__.
    Example:
    ```
    class MyDataset(FiftyOneBaseDataset):
        def __init__(self, dataset : fo.Dataset, transform : callable):
            super().__init__(dataset)
            self.transform = transform
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            fpath = sample["filepath"]
            image = Image.open(fpath)
            image = self.transform(image)
            return image
    ```
    """

    def __init__(self, dataset: fo.Dataset):
        super().__init__()
        self.dataset = dataset

        # dataset cannot be accessed by index but by 'id'
        self.ids = dataset.values("id")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        sample = self.dataset[_id]
        return sample


class HorizonDataset(FiftyOneBaseDataset):
    """Dataset for horizon detection training.
    """

    def __init__(self,
                 dataset: fo.Dataset,
                 transform: callable = None,
                 target_transform: callable = None,
                 ):
        """
        Args:
            dataset (fo.Dataset): FiftyOne dataset
            transform (callable): transform to apply to image and target.
                Example usage: `transform(image=image, keypoints=target)`
            target_transform (callable): transform to apply to target
                Example usage: `target_transform(target, image_w, image_h)`

        Returns:
            image (np.ndarray): HxWxC
            target (np.ndarray): depends on target_transform
        """
        super().__init__(dataset)

        if transform is not None:
            assert callable(transform), "transform must be callable"
        self.transform = transform

        if target_transform is not None:
            assert callable(target_transform), "target_transform must be callable"
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        fpath = sample["filepath"]

        # read image (np.ndarray)
        image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if image.ndim > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w = image.shape[:2]  # numpy has (H,W,C)

        # read points (list of x,y)
        target = sample["ground_truth_pl"]["polylines"][0]["points"][0]
        target = np.array(target).reshape(-1, 2)
        # denormalize points
        target[:, 0] *= image_w
        target[:, 1] *= image_h

        if self.transform:
            # albumentations does not like points in the border
            target = self.shift_points_in_border(target, image.shape[1], image.shape[0])
            augmented = self.transform(image=image, keypoints=target)
            image = augmented["image"]
            image_h, image_w = image.shape[1:]  # tensor has now (C,H,W)
            target = augmented["keypoints"]

        try:
            if self.target_transform:
                target = self.target_transform(target, image_w, image_h)
        except Exception as e:
            print(f"Error in target_transform for sample {fpath}: {e}")
            raise e

        return image, target

    def shift_points_in_border(self, points, img_w, img_h):
        """If points are in the border, shift them inside the image by 1 pixel."""
        points[:, 0] = np.clip(points[:, 0], 1, img_w - 1)
        points[:, 1] = np.clip(points[:, 1], 1, img_h - 1)
        return points
