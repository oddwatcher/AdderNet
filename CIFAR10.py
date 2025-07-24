import os
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
from pathlib import Path
import pickle
from PIL import Image
import tarfile
import hashlib
import urllib.request
from tqdm import tqdm


class CIFAR10:
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        self.root = Path(root)  # Ensure root is a Path object
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []  # Changed from Any to list for clarity
        self.targets = []

        # Load data from individual batch files
        for (
            file_name,
            _,
        ) in downloaded_list:  # Checksum verification can be added here if needed
            file_path = self.root / self.base_folder / file_name
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                # Ensure data is a list of arrays for vstack later
                if isinstance(entry["data"], np.ndarray):
                    self.data.append(entry["data"])
                else:
                    # If it's not an array, try to convert (shouldn't happen with standard CIFAR)
                    self.data.append(np.array(entry["data"]))

                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    # CIFAR-10 uses 'labels', but check for 'fine_labels' just in case
                    self.targets.extend(entry.get("fine_labels", []))

        # Now correctly stack the list of arrays
        if self.data:  # Check if data list is not empty
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose(
                (0, 2, 3, 1)
            )  # convert to HWC (Height, Width, Channels)
        else:
            # Handle case where no data was loaded (should not happen if integrity check passed)
            self.data = np.array([])
            # self.targets would also be empty

        self._load_meta()

    def _load_meta(self) -> None:
        path = self.root / self.base_folder / self.meta["filename"]

        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if index >= len(self):
            raise IndexError(
                f"Index {index} is out of bounds for dataset with {len(self)} items."
            )

        img, target = self.data[index], self.targets[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return (
            len(self.data)
            if self.data is not None and len(self.data) > 0
            else len(self.targets)
        )

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _check_integrity(self) -> bool:

        root = self.root
        base_folder_path = root / self.base_folder
        if not base_folder_path.is_dir():
            return False

        lists_to_check = self.train_list if self.train else self.test_list
        for filename, md5 in lists_to_check:
            fpath = base_folder_path / filename
            if not fpath.is_file():
                return False

        # Check meta file
        meta_fpath = base_folder_path / self.meta["filename"]
        if not meta_fpath.is_file():
            print(f"Missing meta file: {meta_fpath}")  # Optional debug print
            return False

        return True

    def download(self) -> None:
        """Download and extract the CIFAR-10 data if it doesn't exist."""
        root = Path(self.root)
        root.mkdir(parents=True, exist_ok=True)

        base_folder_path = root / self.base_folder
        if base_folder_path.is_dir():
            print("Dataset already downloaded and verified.")
            return

        filename = self.filename
        file_path = root / filename
        if not file_path.is_file():
            print(f"Downloading {self.url} to {file_path}")

            def progress_hook(t):
                last_b = [0]

                def update_to(b=1, bsize=1, tsize=None):
                    if tsize not in (None, 0):
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b
                    if tsize is not None and t.n >= tsize:
                        t.close()

                return update_to

            with tqdm(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
            ) as t:
                urllib.request.urlretrieve(
                    self.url, file_path, reporthook=progress_hook(t)
                )

        print(f"Extracting {file_path}")
        with tarfile.open(file_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = base_folder_path / member.name
                if not (
                    member_path.resolve().is_relative_to(base_folder_path.resolve())
                    or member_path.resolve() == base_folder_path.resolve()
                ):
                    raise Exception(
                        f"Attempted Path Traversal in Tar File: {member.name}"
                    )
            tar.extractall(path=root)

        file_path.unlink()
