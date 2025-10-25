# model/dataset.py
from typing import Callable, Optional, List
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def default_mask_mapper(img_path: str, img_dir: str, mask_dir: str) -> str:
    """
    mask & image has same file name, different folder name
    """
    fname = os.path.basename(img_path)
    return os.path.join(mask_dir, fname)


class PFIBSliceDataset(Dataset):
    """
    - read 2D slices
    - mask shape [H,W], value {0,1}
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transform_pair: Optional[Callable] = None,
        to_tensor: Optional[Callable] = None,
        mask_mapper: Callable = default_mask_mapper,
        grayscale: bool = True,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_pair = transform_pair
        self.to_tensor = to_tensor
        self.mask_mapper = mask_mapper
        self.grayscale = grayscale

        self.img_paths: List[str] = [
            os.path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found under {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _open_img(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return img

    def _open_mask(self, path: str) -> Image.Image:
        # nonzero -> 1
        m = Image.open(path).convert("L")
        return m

    def __getitem__(self, idx: int):
        ip = self.img_paths[idx]
        mp = self.mask_mapper(ip, self.img_dir, self.mask_dir)

        img = self._open_img(ip)
        mask = self._open_mask(mp)

        if self.transform_pair is not None:
            img, mask = self.transform_pair(img, mask)

        # to_tensor
        if self.to_tensor is not None:
            img = self.to_tensor(img)  # [C,H,W] float32
        else:
            from torchvision.transforms.functional import to_tensor
            img = to_tensor(img)

        from torchvision.transforms.functional import pil_to_tensor
        mask = pil_to_tensor(mask).squeeze(0)  # [H,W], uint8
        mask = (mask > 0).long()               # {0,1} -> LongTensor

        return img, mask
