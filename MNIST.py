import numpy as np
from PIL import Image
import os
import gzip
import requests

class MNISTDataset:
    def __init__(self, root, train=True, download=False):
        self.root = root
        self.train = train
        if download:
            self._download()
        self._load_data()

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        urls = {
            'train-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        }
        for filename, url in urls.items():
            gz_path = os.path.join(self.root, filename)
            if not os.path.exists(gz_path):
                print(f"Downloading {filename}...")
                response = requests.get(url)
                with open(gz_path, 'wb') as f:
                    f.write(response.content)
            
            raw_path = gz_path[:-3]
            if not os.path.exists(raw_path):
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(raw_path, 'wb') as f_out:
                        f_out.write(f_in.read())

    def _load_data(self):
        if self.train:
            images_file = os.path.join(self.root, 'train-images-idx3-ubyte')
            labels_file = os.path.join(self.root, 'train-labels-idx1-ubyte')
        else:
            images_file = os.path.join(self.root, 't10k-images-idx3-ubyte')
            labels_file = os.path.join(self.root, 't10k-labels-idx1-ubyte')
        
        with open(images_file, 'rb') as f:
            data = f.read()
            num_images = int.from_bytes(data[4:8], 'big')
            rows = int.from_bytes(data[8:12], 'big')
            cols = int.from_bytes(data[12:16], 'big')
            self.data = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows, cols)
        
        with open(labels_file, 'rb') as f:
            data = f.read()
            num_labels = int.from_bytes(data[4:8], 'big')
            self.targets = np.frombuffer(data[8:], dtype=np.uint8)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index], mode='L')
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)