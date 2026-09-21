import os
import logging
from typing import List, Tuple, Optional

from config import Config 

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dataset_utils.log", mode='a')
    ]
)


def parse_txt(
    txt_path: str,
    images_root: str = Config.ROOT + "\\images"
) -> Tuple[List[str], List[int]]:
    img_paths: List[str] = []
    labels: List[int] = []
    missing_count = 0

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Parsing {os.path.basename(txt_path)}", unit="img"):
            parts = line.strip().split()
            if len(parts) == 2:
                img_rel_path, label_str = parts
                img_full_path = os.path.join(images_root, img_rel_path)
                if os.path.exists(img_full_path):
                    img_paths.append(img_full_path)
                    try:
                        labels.append(int(label_str))
                    except ValueError:
                        logging.warning(f"Invalid label '{label_str}' for image {img_full_path}. Skipping.")
                        img_paths.pop()  # Remove the last added path
                else:
                    missing_count += 1
                    logging.warning(f"Missing image: {img_full_path}")
            else:
                logging.warning(f"Malformed line in {txt_path}: '{line.strip()}'")

    logging.info(f"Loaded {len(img_paths)} images from {txt_path}. Missing files: {missing_count}")
    return img_paths, labels


class IP102Dataset(Dataset):
    """
    Custom Dataset for the IP102 pest dataset.
    """
    def __init__(
        self,
        img_paths: List[str],
        labels: List[int],
        transform: Optional[object] = None
    ):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logging.error(f"File not found during dataset access: {img_path}. Attempting fallback.")
            # Fallback: try next image (circular)
            return self.__getitem__((idx + 1) % len(self.img_paths))
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}. Attempting fallback.")
            return self.__getitem__((idx + 1) % len(self.img_paths))

        if self.transform:
            image = self.transform(image)
        return image, label
