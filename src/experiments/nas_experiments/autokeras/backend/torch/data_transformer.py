import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.nas_experiments.autokeras.constant import Constant


class MultiTransformDataset(Dataset):
    """A class incorporate all transform method into a torch.Dataset class."""

    def __init__(self, dataset, target, compose):
        self.dataset = dataset
        self.target = target
        self.compose = compose

    def __getitem__(self, index):
        feature = self.dataset[index]
        if self.target is None:
            return self.compose(feature)
        return self.compose(feature), self.target[index]

    def __len__(self):
        return len(self.dataset)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """Perform the actual transformation.

from experiments.nas_experiments.autokeras.preprocessor import DataTransformer
        Args:
            img (Tensor): Tensor image of size (C, H, W).

        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DataTransformerMlp(DataTransformer):
    def __init__(self, data):
        super().__init__()
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform_train(self, data, targets=None, batch_size=None):
        data = (data - self.mean) / self.std
        data = np.nan_to_num(data)
        dataset = self._transform([], data, targets)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def transform_test(self, data, target=None, batch_size=None):
        return self.transform_train(data, targets=target, batch_size=batch_size)

    @staticmethod
    def _transform(compose_list, data, targets):
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = torch.Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)