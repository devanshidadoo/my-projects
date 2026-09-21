import logging
from .dataset_utils import parse_txt, IP102Dataset
from torch.utils.data import DataLoader

class DataModule:
    """
    Modular data loader for the IP102 pest dataset.
    Handles parsing, dataset creation, and DataLoader instantiation with logging.
    """
    def __init__(
        self,
        train_txt_path,
        val_txt_path,
        test_txt_path,
        train_transform,
        val_test_transform,
        batch_size,
        num_workers=4
    ):
        self.train_txt_path = train_txt_path
        self.val_txt_path = val_txt_path
        self.test_txt_path = test_txt_path
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._prepare_data()

    def _prepare_data(self):
        logging.info("Parsing dataset splits...")

        self.train_img_paths, self.train_labels = parse_txt(self.train_txt_path)
        self.val_img_paths, self.val_labels = parse_txt(self.val_txt_path)
        self.test_img_paths, self.test_labels = parse_txt(self.test_txt_path)

        logging.info(f"Loaded {len(self.train_img_paths)} training images, "
                     f"{len(self.val_img_paths)} validation images, "
                     f"and {len(self.test_img_paths)} test images.")

        self.train_dataset = IP102Dataset(self.train_img_paths, self.train_labels, transform=self.train_transform)
        self.val_dataset = IP102Dataset(self.val_img_paths, self.val_labels, transform=self.val_test_transform)
        self.test_dataset = IP102Dataset(self.test_img_paths, self.test_labels, transform=self.val_test_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        logging.info("Data loaders created successfully.")

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
