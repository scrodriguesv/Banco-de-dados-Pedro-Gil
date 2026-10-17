import torch, os, cv2
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import albumentations as A


class CustomDataset(Dataset):

    def __init__(self, df_path, src_path):
        '''
        A class for your custom dataset.
        In the init you should do all the preprocessing and data loading
        that should be done only once, such as scaling, one-hot encoding, etc.
        Feel free to create auxiliary methods for this class.

        Args:
            Include here the arguments you need to initialize your dataset.
        '''
        self.df = pd.read_csv(df_path)
        self.num_data = self.df.shape[0]
        self.preprend_path = src_path
    
    def __len__(self): 
        return self.num_data
    
    def __getitem__(self, idx):
        '''
        Must-have method for pytorch dataloader.
        Gets the data and the labels for a given index.
        You should load the data based on the idx and pair it with its labels.
        Here is where you should perform online data augmentation.

        Args:
            idx (int): index of the data. You don't need to worry about it.

        Returns:
            data (_type_): describe the data type.
            labels (_type_): describe the labels type.
        '''
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        row = self.df.iloc[idx]

        image_path = os.path.join(self.preprend_path, row['path_crop'])
        image = Image.open(image_path).convert('RGB') 
        data = transform(image)

        labels = torch.tensor(row['astm_corrosao'])

        return data, labels
    
class SemanticSegmentationDataset(Dataset):
    """
    PyTorch Dataset for image semantic segmentation tasks.

    Expects the following directory structure under ``root_dir``::

        root_dir/
        ├── _classes.csv              # Header row + rows of "id,label"
        └── <specific_path>/
            ├── pixel_values/         # RGB images (.jpg)
            └── bitmaps/              # Single-channel segmentation masks (.png)

    Online augmentation is applied via Albumentations and includes colour
    jitter, horizontal flip, and mild downscaling.  Both image and mask
    receive the same spatial transforms.

    Args:
        root_dir (os.PathLike): Root directory containing ``_classes.csv`` and
            the split sub-folder.
        specific_path (os.PathLike): Sub-folder name for a particular data
            split (e.g. ``"train"`` or ``"val"``).
        feature_extractor (callable): A HuggingFace-compatible feature extractor
            (e.g. ``SegformerImageProcessor``) that accepts a PIL image and a
            PIL mask and returns a dict of tensors.
    """

    def __init__(self, root_dir: os.PathLike, specific_path: os.PathLike, feature_extractor: callable):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        self.specific_path = specific_path

        # Build id → label mapping from the classes CSV (skip header row)
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i, l in enumerate(fid) if i != 0]
        self.id2label = {x[0]: x[1] for x in data}

        # Collect and sort file names so images[i] always pairs with masks[i]
        image_file_names = [f for f in os.listdir(os.path.join(self.root_dir, self.specific_path, 'pixel_values')) if '.jpg' in f]
        mask_file_names  = [f for f in os.listdir(os.path.join(self.root_dir, self.specific_path, 'bitmaps'))       if '.png' in f]

        self.images = sorted(image_file_names)
        self.masks  = sorted(mask_file_names)

        # Albumentations augmentation pipeline applied identically to image and mask
        self.transform = A.Compose([
            A.PlanckianJitter(p=0.50, sampling_method='gaussian'),  # Colour temperature jitter
            A.HorizontalFlip(p=0.50),                               # Random horizontal flip
            A.Downscale(scale_range=(0.9, 0.95), p=0.35),           # Mild resolution downscale
        ], additional_targets={'mask': 'mask'})  # Ensure mask receives matching spatial transforms

    def __len__(self) -> int:
        """Return the total number of image–mask pairs in this split."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """
        Load, augment, and encode the image–mask pair at position ``idx``.

        Steps:
            1. Read image (BGR → RGB) and grayscale mask from disk.
            2. Apply Albumentations augmentations consistently to both.
            3. Convert NumPy arrays to PIL images for the feature extractor.
            4. Run the feature extractor and remove the spurious batch dimension.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Encoded inputs produced by ``self.feature_extractor``, with
                each tensor squeezed to remove the leading batch dimension.
                Typically contains ``"pixel_values"`` and ``"labels"`` keys.
        """
        # Load image and convert from OpenCV BGR to RGB
        image = cv2.imread(os.path.join(self.root_dir, self.specific_path, 'pixel_values', self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load segmentation mask as a single-channel (grayscale) image
        mask = cv2.imread(os.path.join(self.root_dir, self.specific_path, 'bitmaps', self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        # Apply Albumentations transformations to image and mask simultaneously
        augmented = self.transform(image=image, mask=mask)
        image_aug = augmented['image']
        mask_aug  = augmented['mask']

        # Convert augmented NumPy arrays back to PIL for the feature extractor
        image_pil = Image.fromarray(image_aug)
        mask_pil  = Image.fromarray(mask_aug)

        # Encode image and mask; result tensors have a leading batch dimension of 1
        encoded_inputs = self.feature_extractor(image_pil, mask_pil, return_tensors="pt")

        # Squeeze out the batch dimension added by the feature extractor
        for k in encoded_inputs:
            encoded_inputs[k].squeeze_()

        return encoded_inputs