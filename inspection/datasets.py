import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

class AgeGenderDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing age and gender data from multiple sources (IMDB, WIKI, Adience, MORPH).
    """
    def __init__(self, config, dataset=["IMDB", "WIKI", "Adience", "MORPH"], transform=None):
        """
        Initializes the AgeGenderDataset with configuration, dataset sources, and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            dataset (list): List of dataset sources to include.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.root_path = config.age_gender_data_path  # Root path for age and gender data
        self.image_paths = []  # List to store image paths
        self.labels = []  # List to store labels (age and gender)
        self.suffix = ""  # Suffix for label files
        self.sample_num = config.num_image // config.recognition_bz * config.age_gender_bz  # Number of samples to load

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        # Weights for different datasets
        self.weight = dict()
        self.weight["Adience"] = 0.25
        self.weight["MORPH"] = 0.25
        self.weight["IMDB+WIKI"] = 0.5

        # Load data from Adience dataset
        flag = True
        self.labels_1 = []
        while flag:
            if "Adience" in dataset and flag:
                self.Adience_path = os.path.join(self.root_path, "Adience")
                self.Adience_data = os.path.join(self.Adience_path, "data")
                self.Adience_label = os.path.join(self.Adience_path, ("label" + self.suffix + ".txt"))

                with open(self.Adience_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:
                        image_path = os.path.join(self.Adience_data, line[0])
                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])
                            if line[2] == "1.0":
                                label[1] = 1
                            self.labels_1.append(label)

                    if len(self.labels_1) >= self.sample_num * self.weight["Adience"]:
                        flag = False
                        break

        # Load data from MORPH dataset
        flag = True
        self.labels_2 = []
        while flag:
            if "MORPH" in dataset and flag:
                self.MORPH_path = os.path.join(self.root_path, "MORPH")
                self.MORPH_data = os.path.join(self.MORPH_path, "data")
                self.MORPH_label = os.path.join(self.MORPH_path, ("label" + self.suffix + ".txt"))

                with open(self.MORPH_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:
                        image_path = os.path.join(self.MORPH_data, line[0])
                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])
                            if line[2] == "1.0":
                                label[1] = 1
                            self.labels_2.append(label)

                    if len(self.labels_2) >= self.sample_num * self.weight["MORPH"]:
                        flag = False
                        break

        # Combine labels from Adience and MORPH
        self.labels.extend(self.labels_1)
        self.labels.extend(self.labels_2)

        # Load data from WIKI and IMDB datasets
        flag = True
        while flag:
            if "WIKI" in dataset and flag:
                self.WIKI_path = os.path.join(self.root_path, "WIKI")
                self.WIKI_data = os.path.join(self.WIKI_path, "data")
                self.WIKI_label = os.path.join(self.WIKI_path, ("label" + self.suffix + ".txt"))

                with open(self.WIKI_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:
                        image_path = os.path.join(self.WIKI_data, line[0])
                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])
                            if line[2] == "1.0":
                                label[1] = 1
                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

            if "IMDB" in dataset and flag:
                self.IMDB_path = os.path.join(self.root_path, "IMDB")
                self.IMDB_data = os.path.join(self.IMDB_path, "data")
                self.IMDB_label = os.path.join(self.IMDB_path, ("label" + self.suffix + ".txt"))
                with open(self.IMDB_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:
                        image_path = os.path.join(self.IMDB_data, line[0])
                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])
                            if line[2] == "1.0":
                                label[1] = 1
                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a list.
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.labels)

class CelebADataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing CelebA data.
    """
    def __init__(self, config, choose, transform=None):
        """
        Initializes the CelebADataset with configuration, dataset split, and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            choose (str): Dataset split to use ("train", "val", "test").
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.image_names = []  # List to store image names
        self.labels = []  # List to store labels
        random.seed(config.seed)  # Set random seed for reproducibility

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        # Load training data
        if choose == "train":
            self.CelebA_data = config.CelebA_train_data
            self.CelebA_label = config.CelebA_train_label
            self.sample_num = config.num_image // config.recognition_bz * config.CelebA_bz

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(1, len(data)):
                    r = random.random()
                    if r < 0.5:
                        line = data[i].split()
                        self.image_names.append(line[0])
                        label = [0 for j in range(40)]
                        for j in range(1, 41):
                            if line[j] == "1":
                                label[j-1] = 1
                        self.labels.append(label)
                        if len(self.labels) == self.sample_num:
                            flag = False
                            break

        # Load validation data
        elif choose == "val":
            self.CelebA_data = config.CelebA_val_data
            self.CelebA_label = config.CelebA_val_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1
                self.labels.append(label)

        # Load test data
        elif choose == "test":
            self.CelebA_data = config.CelebA_test_data
            self.CelebA_label = config.CelebA_test_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1
                self.labels.append(label)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a list.
        """
        img_path = os.path.join(self.CelebA_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.image_names)

class ExpressionDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing expression data from RAF and AffectNet datasets.
    """
    def __init__(self, config, transform=None):
        """
        Initializes the ExpressionDataset with configuration and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.image_paths = []  # List to store image paths
        self.labels = []  # List to store labels

        self.RAF_data = config.RAF_data
        self.RAF_label = config.RAF_label
        self.AffectNet_data = config.AffectNet_data
        self.AffectNet_label = config.AffectNet_label

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        self.sample_num = config.num_image // config.recognition_bz * config.expression_bz

        # Weights for different datasets
        self.weight = dict()
        self.weight["RAF"] = 0.5
        self.weight["AffectNet"] = 0.5

        # Load data from AffectNet dataset
        with open(self.AffectNet_label, "r") as f:
            data = f.readlines()

        flag = True
        while flag:
            for i in range(1, len(data)):
                line = data[i].split()
                image_path = os.path.join(self.AffectNet_data, line[0])
                self.image_paths.append(image_path)
                self.labels.append(int(line[2]))
                if len(self.labels) >= self.sample_num * self.weight["AffectNet"]:
                    flag = False
                    break

        # Load data from RAF dataset
        with open(self.RAF_label, "r") as f:
            data = f.readlines()

        flag = True
        while flag:
            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")
                image_name = line[0]
                sample_temp = image_name.split("_")[0]
                if sample_temp == "train":
                    image_path = os.path.join(self.RAF_data, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]) - 1)
                if len(self.labels) == self.sample_num:
                    flag = False
                    break

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.labels)

class RAFDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing RAF dataset.
    """
    def __init__(self, config, choose, transform=None):
        """
        Initializes the RAFDataset with configuration, dataset split, and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            choose (str): Dataset split to use ("train", "test").
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.image_names = []  # List to store image names
        self.labels = []  # List to store labels

        self.RAF_data = config.RAF_data
        self.RAF_label = config.RAF_label

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        self.train = True if choose == "train" else False

        # Load training data
        if choose == "train":
            self.sample_num = config.num_image // config.recognition_bz * config.RAF_bz
            with open(self.RAF_label, "r") as f:
                data = f.readlines()
            flag = True
            while flag:
                for i in range(0, len(data)):
                    line = data[i].strip('\n').split(" ")
                    image_name = line[0]
                    sample_temp = image_name.split("_")[0]
                    if self.train and sample_temp == "train":
                        self.image_names.append(image_name)
                        self.labels.append(int(line[1]) - 1)
                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        # Load test data
        elif choose == "test":
            with open(self.RAF_label, "r") as f:
                data = f.readlines()
            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")
                image_name = line[0]
                sample_temp = image_name.split("_")[0]
                if not self.train and sample_temp == "test":
                    self.image_names.append(image_name)
                    self.labels.append(int(line[1]) - 1)

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        img_path = os.path.join(self.RAF_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.image_names)

class FGnetDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing FGnet dataset.
    """
    def __init__(self, config, choose="all", id=0, transform=None):
        """
        Initializes the FGnetDataset with configuration, dataset split, and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            choose (str): Dataset split to use ("all", "9_fold", "1_fold", "remove_one").
            id (int): Index for leave-one-out validation.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.FGnet_data = config.FGnet_data
        self.FGnet_label = config.FGnet_label
        self.leave_out_file_name = ""  # File name for leave-one-out validation

        self.image_names = []  # List to store image names
        self.labels = []  # List to store labels

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        with open(self.FGnet_label, "r") as f:
            data = f.readlines()

        cut = len(data) // 10 * 9  # Index for 90-10 split

        # Load data based on the chosen split
        if choose == "remove_one":
            for i in range(1, len(data)):
                line = data[i].split()
                if i == id:
                    self.leave_out_file = os.path.join(self.FGnet_data, line[0])
                else:
                    self.image_names.append(line[0])
                    label = np.asarray([float(line[1])])
                    self.labels.append(label)
        elif choose == "all":
            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "9_fold":
            for i in range(1, cut):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "1_fold":
            for i in range(cut, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a tensor.
        """
        img_path = os.path.join(self.FGnet_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.image_names)

    def get_leave_out_file_name(self):
        """
        Returns the file name for leave-one-out validation.
        
        Returns:
            str: File name.
        """
        return self.leave_out_file_name

class LAPDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and preprocessing LAP dataset.
    """
    def __init__(self, config, choose="", transform=None):
        """
        Initializes the LAPDataset with configuration, dataset split, and optional transform.
        
        Args:
            config: Configuration object containing paths and parameters.
            choose (str): Dataset split to use ("train", "test").
            transform (callable, optional): Optional transform to apply to the data.
        """
        if choose == "train":
            self.LAP_data = config.LAP_train_data
            self.LAP_label = config.LAP_train_label
        elif choose == "test":
            self.LAP_data = config.LAP_test_data
            self.LAP_label = config.LAP_test_label

        self.image_names = []  # List to store image names
        self.labels = []  # List to store labels

        # Set default transform if not provided
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),  # Resize images
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        with open(self.LAP_label, "r") as f:
            data = f.readlines()

        for i in range(0, len(data)):
            line = data[i].split(";")
            self.image_names.append(line[0])
            label = [np.asarray([float(line[1])]), np.asarray([float(line[2])])]
            self.labels.append(label)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a list of tensors.
        """
        img_path = os.path.join(self.LAP_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]
        return img, label

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: Number of items.
        """
        return len(self.image_names)