""" Module to download Imagenette and generate pytorch dataset """

import os
import requests
import tarfile
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def download_imagenette(url, local_path="./"):
    """
    Args:
        url (_type_): url where the imagenette file is hosted.
        local_path (str, optional): Path where you want to host
                                    the images dataset. Defaults to "./".
    """
    # Select the name of the file to download
    url_path_file = url.split('/')[-1]

    # Default original folder name
    foldername = os.path.splitext(url_path_file)[0]
    final_path = os.path.join(local_path, foldername)
    if not os.path.exists(final_path):
        # Download the file
        if not os.path.exists(url_path_file):
            print("Downloading", url, "...")
            response = requests.get(url)
            open(url_path_file, "wb").write(response.content)
        # Extract files
        print("Starting", url_path_file, "files extraction...")
        file = tarfile.open(url_path_file)
        file.extractall(path=local_path)
        file.close()
        print("Files extraction completed in:", final_path)
        # Remove downloaded file
        os.remove(url_path_file)

    else:
        print(foldername, "folder already exists.")
        print("You can find it in:", final_path)


class Imagenette(Dataset):
    def __init__(self,
                 annotations_file,
                 img_dir,
                 numcol_labels=1,
                 train=True,
                 shuffle=True,
                 transform=None,
                 target_transform=None,
                 random_state=None):
        """Imagenette Dataset.

        Args:
            annotations_file (string): Root directory of the CSV file that
                                stores the labels and path information.
            img_dir (string): Root directory of the imagenette dataset.
            numcol_labels (int, optional): Index of the column in the
                                annotations_file that represents the labels.
                                Defaults to 1.
            train (bool, optional): If True, creates dataset from training set;
                                otherwise creates from test set.
                                Defaults to True.
            shuffle (bool, optional): If True, the data is shuffled;
                                otherwise not.
                                Defaults to True.
            transform (callable, optional): A function/transformation that
                                takes a tensor image and returns a transformed
                                version. For example, transforms.RandomCrop.
                                Defaults to None.
            target_transform (callable, optional): A function/transformation
                                that takes a integer label and returns a
                                transformed version.
                                Defaults to None.
            random_state (int, optional): Seed for random number generator.
        """

        self.info_csv = pd.read_csv(annotations_file)
        self.numcol_labels = numcol_labels
        self.img_dir = img_dir

        if train:
            self.info_csv = self.info_csv[~ self.info_csv['is_valid']]
            self.info_csv = self.info_csv.reset_index(drop=True)
        else:
            self.info_csv = self.info_csv[self.info_csv['is_valid']]
            self.info_csv = self.info_csv.reset_index(drop=True)

        if shuffle:
            self.info_csv = self.info_csv.sample(frac=1,
                                                 random_state=random_state)
            self.info_csv = self.info_csv.reset_index(drop=True)

        self.transform = transform
        self.target_transform = target_transform

        self.lbl_dict = dict(
            n01440764='tench',
            n02102040='English springer',
            n02979186='cassette player',
            n03000684='chain saw',
            n03028079='church',
            n03394916='French horn',
            n03417042='garbage truck',
            n03425413='gas pump',
            n03445777='golf ball',
            n03888257='parachute'
        )
        # Transformation to convert original labels to integers
        self.old_lbs = sorted(self.info_csv.iloc[:, 1].unique())
        self.cat_to_num = dict(zip(self.old_lbs, range(len(self.old_lbs))))
        col_num = self.numcol_labels
        self.info_csv.iloc[:, col_num] = self.info_csv.iloc[:, col_num].apply(
            lambda x: self.cat_to_num[x])

    def __len__(self):
        return len(self.info_csv)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.info_csv.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.info_csv.iloc[idx, self.numcol_labels]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
