# model/dataset.py
from typing import Callable, Optional, List
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor as tv_to_tensor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def default_mask_mapper(img_path: Path, img_root: Path, mask_root: Path) -> Path:
    rel = img_path.relative_to(img_root)
    return mask_root / rel


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
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform_pair = transform_pair
        self.to_tensor = to_tensor
        self.mask_mapper = mask_mapper
        self.grayscale = grayscale

        self.img_paths = sorted([
            p for p in self.img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        if not self.img_paths:
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
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int):
        ip = self.img_paths[idx]
        mp = self.mask_mapper(ip, self.img_dir, self.mask_dir)

        img = self._open_img(ip)
        mask = self._open_mask(mp)
        
        # check shape
        if img.size != mask.size:
            raise ValueError(
            f"[PFIBDataset] Image/mask size mismatch for '{ip}': "
            f"img={img.size}, mask={mask.size}"
        )

        if self.transform_pair is not None:
            img, mask = self.transform_pair(img, mask)

        # to_tensor
        if self.to_tensor is not None:
            img_tensor = self.to_tensor(img)
        else:
            img_tensor = tv_to_tensor(img)  # [C,H,W], float32

        # mask -> tensor {0,1} long
        mask_tensor = pil_to_tensor(mask).squeeze(0)  # [H,W], uint8
        mask_tensor = (mask_tensor > 0).long()

        return img_tensor, mask_tensor
