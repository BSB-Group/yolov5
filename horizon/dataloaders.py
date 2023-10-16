import numpy as np
import fiftyone as fo
import albumentations as A
from torch.utils.data import Dataset
from utils.dataloaders import imread_16bit_compatible
from horizon.utils import points_to_pitch_theta, points_to_hough


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
                 transform: A.Compose = None,
                 target_format: str = "points",
                 augment16bit: bool = False,
                 ):
        """
        Args:
            dataset (fo.Dataset): FiftyOne dataset
            transform (A.Compose): albumentations transform
            target_format (str): possible values: "points", "hough", "pitch_theta"

        Returns:
            image (np.ndarray): HxWxC
            target (np.ndarray): depends on target_format
        """
        super().__init__(dataset)

        if transform is not None:
            assert isinstance(transform, A.Compose)
        self.transform = transform

        assert target_format in ["points", "hough", "pitch_theta"]
        self.target_format = target_format

        assert isinstance(augment16bit, bool)
        self.augment16bit = augment16bit

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        fpath = sample["filepath"]

        # read image (np.ndarray)
        image = imread_16bit_compatible(fpath, augment16=self.augment16bit)
        image_h, image_w = image.shape[:2] # numpy has (H,W,C)

        # read points (list of x,y)
        target = sample["ground_truth_pl"]["polylines"][0]["points"][0]
        target = np.array(target).reshape(-1, 2)
        target[:, 0] *= image_w
        target[:, 1] *= image_h

        if self.transform:
            # albumentations does not like points in the border
            target = self.shift_points_in_border(target, image.shape[1], image.shape[0])
            augmented = self.transform(image=image, keypoints=target)
            image = augmented["image"]
            image_h, image_w = image.shape[1:] # tensor has now (C,H,W)
            target = augmented["keypoints"]

        if self.target_format == "points":
            target = np.array(target).flatten()  # x1, y1, x2, y2

        elif self.target_format == "pitch_theta":
            # normalize points
            target = np.array(target)
            target[:, 0] /= image_w
            target[:, 1] /= image_h

            # convert to pitch, theta
            x1, y1, x2, y2 = target.flatten()
            pitch, theta = points_to_pitch_theta(x1, y1, x2, y2)
            target = np.array([pitch, theta])

        elif self.target_format == "hough":
            x1, y1, x2, y2 = np.array(target).flatten()
            rho, theta = points_to_hough(x1, y1, x2, y2)
            target = np.array([rho, theta])

        return image, target

    def shift_points_in_border(self, points, img_w, img_h):
        """If points are in the border, shift them inside the image by 1 pixel."""
        points[:, 0] = np.clip(points[:, 0], 1, img_w - 1)
        points[:, 1] = np.clip(points[:, 1], 1, img_h - 1)
        return points
