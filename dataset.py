import pandas as pd
import ast
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch

from processing import data_augmentation, preprocessing, enriched_data_augmentation


class MyDataset(Dataset):
    def __init__(
        self,
        args,
        dataset: pd.DataFrame,
        DA,
        device: str = "cuda",
    ):
        """
        Custom Pytorch Dataset class for deep learning.

        Args:
            args (Namespace): job arguments from the parser, see config.py.
            dataset (pd.DataFrame): DataFrame containing dataset information (series_path, mask_path, label, clinics, ...).
            DA (float): Probability of applying data augmentation.
            device (str): "cuda" or "cpu".
        """

        self.args = args
        self.DA = DA
        self.data = dataset
        self.device = device

    def __len__(self) -> int:
        """
        Return the number of rows (==patients==series) in the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Use manifest infos to load, preprocess, data_augment and return image and mask(s).
        """

        serie = sitk.ReadImage(self.data["img_path"].iloc[idx])
        mask_tum = sitk.ReadImage(self.data["mask_tum"].iloc[idx])
        box_indices = ast.literal_eval(self.data["bounding_boxes"].iloc[idx])
        patient_id = self.data["patient_id"].iloc[idx]

        img, original_size, original_orientation, crop_size, _ = preprocessing(
            serie,
            size=(
                self.args.image_size_z,
                self.args.image_size_x,
                self.args.image_size_y,
            ),
            bounding_box=box_indices,
            clip=self.args.clip,
        )

        mask_tum = preprocessing(
            mask_tum,
            size=(
                self.args.image_size_z,
                self.args.image_size_x,
                self.args.image_size_y,
            ),
            bounding_box=box_indices,
            is_mask=True,
        )[0]

        if not self.args.tumor_only:
            mask_lung = sitk.ReadImage(self.data["mask_lung"].iloc[idx])

            mask_lung = preprocessing(
                mask_lung,
                size=(
                    self.args.image_size_z,
                    self.args.image_size_x,
                    self.args.image_size_y,
                ),
                bounding_box=box_indices,
                is_mask=True,
            )[0]

            if self.DA:
                if self.DA == "baseline":
                    img, mask_tum, mask_lung = data_augmentation(
                        img, mask_tum, mask_lung
                    )

            return img, mask_tum, patient_id, mask_lung

        if self.DA:
            if self.DA == "baseline":
                img, mask_tum = data_augmentation(img, mask_tum)
            if self.DA == "enriched":
                img, mask_tum = enriched_data_augmentation(img, mask_tum[None, :])
            else:
                pass

        return img.type(torch.float32), mask_tum.type(torch.float32), patient_id
