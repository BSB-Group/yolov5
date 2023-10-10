import numpy as np
import fiftyone as fo
from torch.utils.data import Dataset
from PIL import Image

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

    def __init__(self, dataset: fo.Dataset,
                 transforms: callable = None
                 ):
        super().__init__(dataset)
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        fpath = sample["filepath"]

        image = Image.open(fpath).convert('RGB') # PIL image (LetterBox augmentation in transform expects 3-channel image)
        image = np.array(image)

        target = sample["ground_truth_pl"]["polylines"][0]["points"][0] # list

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
