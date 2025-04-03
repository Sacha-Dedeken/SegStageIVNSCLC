"""This module contains the transformations applied to the images and masks."""

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import (
    BorderPad,
    Compose,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandRotated,
    RemoveSmallObjects,
    Resize,
    SpatialCrop,
    RandFlipd,
    RandSpatialCropd,
    Resized,
)
from monai.utils import InterpolateMode
from torch import Tensor
from torch.nn import Sigmoid


def enriched_data_augmentation(img, seg1) -> tuple[Tensor, Tensor]:
    """Data augmentation on training data
    Add realistic perturbations to the images used to train the model
    in order to increase its robustness.

    Args:
        img (Tensor): original image
        seg (Tensor): original segmentation

    Returns:
        Tensor: modified image
        Tensor: modified segmentation
    """
    img_shape = img.shape
    trf = Compose(
        [
            RandGaussianNoised(keys=["img"], prob=0.2, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.5, 1.5)),
            RandRotated(
                keys=["img", "seg1"],
                prob=0.1,
                range_x=0.15,
                mode=["bilinear", "nearest"],
            ),
            RandRotated(
                keys=["img", "seg1"],
                prob=0.1,
                range_y=0.15,
                mode=["bilinear", "nearest"],
            ),
            RandRotated(
                keys=["img", "seg1"],
                prob=0.1,
                range_z=0.15,
                mode=["bilinear", "nearest"],
            ),
            RandSpatialCropd(
                keys=["img", "seg1"],
                roi_size=[img_shape[1] - 10, img_shape[2] - 10, img_shape[3] - 10],
                random_size=True,
            ),
            Resized(
                keys=["img", "seg1"],
                spatial_size=img_shape[1:],
                mode=["bilinear", "nearest"],
            ),
        ],
    )

    out_dict = trf({"img": img, "seg1": seg1})
    return out_dict["img"], out_dict["seg1"][0]


def data_augmentation(img, seg1, seg2=False) -> tuple[Tensor, Tensor, Tensor]:
    """Data augmentation on training data
    Add realistic perturbations to the images used to train the model
    in order to increase its robustness.

    Args:
        img (Tensor): original image
        seg (Tensor): original segmentation

    Returns:
        Tensor: modified image
        Tensor: modified segmentation
    """
    if not torch.is_tensor(seg2):
        trf = Compose(
            [
                RandGaussianNoised(keys=["img"], prob=0.2, mean=0, std=0.1),
                RandAdjustContrastd(
                    keys=["img"], prob=0.2, gamma=(0.5, 1.5)
                ),  # Randomly changes image intensity with gamma transform
                RandRotated(
                    keys=["img", "seg1"],
                    prob=0.1,
                    range_x=0.1,
                    mode=["bilinear", "nearest"],
                ),
                RandRotated(
                    keys=["img", "seg1"],
                    prob=0.1,
                    range_y=0.1,
                    mode=["bilinear", "nearest"],
                ),
                RandRotated(
                    keys=["img", "seg1"],
                    prob=0.1,
                    range_z=0.1,
                    mode=["bilinear", "nearest"],
                ),
            ],
        )

        out_dict = trf({"img": img, "seg1": seg1})
        return out_dict["img"], out_dict["seg1"]

    else:
        trf = Compose(
            [
                RandGaussianNoised(keys=["img"], prob=0.2, mean=0, std=0.1),
                RandAdjustContrastd(
                    keys=["img"], prob=0.2, gamma=(0.5, 1.5)
                ),  # Randomly changes image intensity with gamma transform
                RandRotated(
                    keys=["img", "seg1", "seg2"],
                    prob=0.1,
                    range_x=0.1,
                    mode=["bilinear", "nearest", "nearest"],
                ),
                RandRotated(
                    keys=["img", "seg1", "seg2"],
                    prob=0.1,
                    range_y=0.1,
                    mode=["bilinear", "nearest", "nearest"],
                ),
                RandRotated(
                    keys=["img", "seg1", "seg2"],
                    prob=0.1,
                    range_z=0.1,
                    mode=["bilinear", "nearest", "nearest"],
                ),
            ],
        )

        out_dict = trf({"img": img, "seg1": seg1, "seg2": seg2})
        return out_dict["img"], out_dict["seg1"], out_dict["seg2"]


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize image between 0 and 1

    Args:
        x (torch.Tensor]): Input Tensor

    Returns:
        [tensor]: Normalized image
    """
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def preprocessing(
    img: sitk.Image,
    size: tuple[int, int, int],
    bounding_box: tuple[int, int, int, int, int, int] = None,
    *,
    is_mask: bool = False,
    clip: str = "abdo",
) -> tuple[Tensor, tuple[int, int, int], str, tuple[int, int, int]]:
    """Preprocessing applied to every image/mask before training and inference.

    Args:
        img (sitk.Image): image to preprocess
        size (int, int, int): size desired for the output
        bounding_box (optional)(int, int, int, int, int, int): bounding box to crop on
                      if no bounding box is given, the image is not cropped. Defaults to None.
        is_mask (bool): flag to differentiate between image and mask. Defaults to image.

    Returns:
        Tensor: the preprocessed image (tensor with shape CHWD)
        tuple[int, int, int]: original size
        string: original orientation
        tuple[int, int, int]: crop size
    """

    img, original_orientation = change_orientation(img, "LPS")

    tensor = Tensor(sitk.GetArrayFromImage(img))
    original_size = tensor.size()
    tensor = tensor.unsqueeze(0)

    if not is_mask:
        if clip == "abdo":
            tensor = torch.clip(tensor, -100, 220)
            not_black = (tensor.squeeze() > -100).float()
        if clip == "lung":
            tensor = torch.clip(tensor, -1350, 150)
            not_black = (tensor.squeeze() > -1000).float()
        else:
            not_black = (tensor.squeeze() > -1000).float()
    else:
        not_black = None

    if bounding_box is not None and np.sum(bounding_box) != 0:
        tensor = SpatialCrop(roi_start=bounding_box[:3], roi_end=bounding_box[3:])(
            tensor,
        )
    crop_size = tensor.squeeze().size()

    tensor = Resize(
        spatial_size=size,
        mode=InterpolateMode.NEAREST_EXACT if is_mask else InterpolateMode.AREA,
    )(tensor)

    if not is_mask:
        tensor = normalize(tensor)
    else:
        tensor = tensor.squeeze()

    return tensor, original_size, original_orientation, crop_size, not_black


def postprocessing(  # noqa: PLR0913
    pred: torch.Tensor,
    original_size: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    original_orientation: str,
    bounding_box: list[int] | None = None,
    not_black: torch.Tensor | None = None,
    thres: float = 0.5,
) -> np.ndarray:
    """Post-processing applied to every test image/mask after inference.

    Args:
        original_size (tuple[int, int, int]): original size
        crop_size (tuple[int, int, int]): size of the bounding box around the lungs where cropping took place
        original_orientation (str): original image orientation
        bounding_box (list[int] | None, optional): bounding box around the lungs where cropping took place. Defaults to None.
        thres (float, optional): threshold for the generated masks. Defaults to 0.5.

    Returns:
        torch.Tensor: the predicted segmentation mask
    """

    pred = Resize(spatial_size=crop_size, mode="trilinear")(pred)
    pred = (Sigmoid()(pred) > thres).float()
    if bounding_box is not None and np.sum(bounding_box) != 0:
        pred = BorderPad(  # Pad the bounding box with 0-s to get back to the original image size
            [
                bounding_box[0],
                original_size[0] - bounding_box[3],
                bounding_box[1],
                original_size[1] - bounding_box[4],
                bounding_box[2],
                original_size[2] - bounding_box[5],
            ],
        )(
            pred
        )

    # if not_black is not None:
    #    pred *= not_black.to(pred.get_device())

    # pred = RemoveSmallObjects(min_size=300)(pred)
    pred = pred.squeeze().cpu().numpy()

    if original_orientation != "unchanged":
        pred_sitk = sitk.GetImageFromArray(pred)
        pred_sitk.SetDirection(  # assign in header the vector version of LPS (Left Posterior Superior)
            sitk.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation("LPS")
        )
        pred_sitk = sitk.DICOMOrient(
            pred_sitk, original_orientation
        )  # Rotate back to original orientation
        pred = sitk.GetArrayFromImage(pred_sitk)

    return pred


def change_orientation(
    img: sitk.Image,
    model_orientation: str,
) -> tuple[sitk.Image, str]:
    """Change the orientation of a sitk image.

    Args:
        img: image whose orientation is to be changed (sitk image)
        model_orientation: Image orientation (str)
             - LPS corresponds to axial from head to feet
             - LIP corresponds to coronal
             - PIR corresponds to sagittal
            see https://www.slicer.org/wiki/Coordinate_systems for more details

    Returns:
        img: img with the right orientation for the model (sitk image)
        original_orientation: original orientation, to be able to return to the original orientation  (str)
    """
    original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection(),
    )

    if original_orientation == model_orientation:
        return img, "unchanged"

    return sitk.DICOMOrient(img, model_orientation), original_orientation
