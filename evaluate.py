import torch
import numpy as np
import pandas as pd
import ast
import SimpleITK as sitk
import warnings
import os
import json
from datetime import datetime
from pytz import timezone
import itertools

from processing import preprocessing, postprocessing
from utils import enable_dropout

from torchmetrics.classification import BinaryJaccardIndex, Dice

from commonlib.requesting import create_requester, manage_segmentation_conversion

from commonlib.bounding_box.crop import (
    load_model_bb,
    compute_pred_lung,
    get_bb_from_mask,
)
from commonlib.reader import read_dicom


def evaluate_model(model, df, args):
    """
    Evaluate predictions from trained model on dataframe, save nrrd in storage and returns scores and paths.
    """

    preds_paths_list = list()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("START EVALUATION", flush=True)
    
    df = df.rename({"series_uid":"series_uids"},axis=1)

    if args.job_mode == "evaluate_from_dicom":
        with open(
            os.path.join(
                args.exp_folder, args.model_version, "Inputs/list_of_preds.json"
            )
        ) as f:
            J = json.load(f)

        df = pd.DataFrame(
            {
                "patient_id": [x["patient_id"] for x in J["list_of_prediction_inputs"]],
                "study_date": [
                    x["series"][0]["study_date"] for x in J["list_of_prediction_inputs"]
                ],
                "series_name": [
                    x["series"][0]["series_name"]
                    for x in J["list_of_prediction_inputs"]
                ],
                "series_uid": [
                    x["series"][0]["series_uid"] for x in J["list_of_prediction_inputs"]
                ],
            }
        )

        for i, el in enumerate(J["list_of_prediction_inputs"]):
            series_paths = [
                os.path.join(args.dcm_dir, x) for x in el["series"][0]["series_paths"]
            ]
            decryption_key = el["series"][0]["decryption_key"]

            img_sitk = read_dicom(series_paths, decryption_key)

            lung_model = load_model_bb()

            bounding_box = get_bb_from_mask(compute_pred_lung(img_sitk, lung_model))

            patient_id = el["patient_id"]

            study_date = el["series"][0]["study_date"]

            series_uid = el["series"][0]["series_uid"]

            prediction(
                model,
                img_sitk,
                bounding_box,
                df,
                patient_id,
                study_date,
                series_uid,
                preds_paths_list,
                args,
                i,
            )

    else:
        for index, row in df.iterrows():
            
            print("start for loop")
            
            img_sitk = sitk.ReadImage(row["img_path"])
            bounding_box = ast.literal_eval(row["bounding_boxes"])
            
            print("img obtained")

            mask_tum = torch.tensor(
                sitk.GetArrayFromImage(sitk.ReadImage(row["mask_tum"]))
            )  # get ground-truth tumor mask
            
            print("mask obtained")

            prediction(
                model,
                img_sitk,
                bounding_box,
                df,
                row["patient_id"],
                row["study_date"],
                row["series_uids"],
                preds_paths_list,
                args,
                index,
                mask_tum,
            )

    csv_path = os.path.join(
        args.output_path,
        args.model_version,
        "grid_search",
        args.suffix,
        "fold_" + str(args.fold),
    )
    os.makedirs(csv_path, exist_ok=True)

    df.to_csv("outputs/test_metrics.csv", index=False)
    df.to_csv(os.path.join(csv_path, "test.csv"), index=False)

    print(
        f"\nEnd of evaluation, the mean tumor dice on this test set/fold is {np.mean(df['dice_phd'])}"
    )

    with open(
        os.path.join(args.output_path, args.model_version, args.preds2convert_path), "w"
    ) as json_file:
        json.dump(preds_paths_list, json_file)

    return preds_paths_list


def prediction(
    model,
    img_sitk,
    bounding_box,
    df,
    patient_id,
    study_date,
    series_uid,
    preds_paths_list,
    args,
    index,
    mask_tum=None,
):
    """ """

    threshold_img_size = 100 * 512 * 512
    threshold_pred_tum = 0.5

    img, original_size, original_orientation, crop_size, not_black = preprocessing(
        img=img_sitk,
        size=(args.image_size_x, args.image_size_y, args.image_size_z),
        bounding_box=bounding_box,
        clip=args.clip,
    )

    img = img.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("try", patient_id)
    
    preds = torch.zeros((fp_numbers, args.image_size_x, args.image_size_y, args.image_size_z))

    model.eval()
    with torch.no_grad():
        enable_dropout(model)
        for ifp in range(args.forward_passes_MCDO):
            pred = model(img.unsqueeze(0))
            preds[ifp] = pred[0,0].cpu()
    
    pred = preds.mean(axis=0).unsqueeze(0)
    
    # compute 10 dices between each pair of preds
    interdices = list()
    for i, j in itertools.combinations(range(preds.shape[0]), 2):
        
        first_pred = postprocessing(
            pred=preds[i].unsqueeze(0),
            original_size=original_size,
            crop_size=crop_size,
            original_orientation=original_orientation,
            bounding_box=bounding_box,
            thres=threshold_pred_tum,
            not_black=not_black,
        )
        first_pred = torch.tensor(first_pred)
        
        second_pred = postprocessing(
            pred=preds[j].unsqueeze(0),
            original_size=original_size,
            crop_size=crop_size,
            original_orientation=original_orientation,
            bounding_box=bounding_box,
            thres=threshold_pred_tum,
            not_black=not_black,
        )
        second_pred = torch.tensor(second_pred).int()
        
        interdice = compute_volume_metrics(
            y_pred=first_pred,
            y_true=second_pred,
            device=torch.device(
                "cuda"
                if torch.cuda.is_available()
                and np.prod(img_sitk.GetSize()) < threshold_img_size
                else "cpu"
            ),
        )["dice"]
        
        interdices.append(interdice)
        
    pred = postprocessing(
        pred=pred,
        original_size=original_size,
        crop_size=crop_size,
        original_orientation=original_orientation,
        bounding_box=bounding_box,
        thres=threshold_pred_tum,
        not_black=not_black,
    )

    print(f"- pred {patient_id} without issue")

    if args.save_pred:
        dir_path = os.path.join(
            args.output_path,
            args.model_version,
            args.suffix + "_Predictions",
            patient_id,
            study_date,
            series_uid,
        )
        os.makedirs(dir_path, exist_ok=True)

        pred_sitk = sitk.GetImageFromArray(pred)
        pred_sitk.CopyInformation(img_sitk)

        copy_conversion_info(pred_sitk, series_uid, args.model_version, args.suffix)

        sitk.WriteImage(
            pred_sitk, os.path.join(dir_path, f"tum-pred-{args.suffix}.nrrd")
        )

        preds_paths_list.append(
            os.path.join(
                "phd_robustness",
                args.model_version,
                args.suffix + "_Predictions",
                patient_id,
                study_date,
                series_uid,
                f"tum-pred-{args.suffix}.nrrd",
            )
        )

    pred = torch.tensor(pred)

    if torch.is_tensor(mask_tum):
        
        print("y_pred shape and type:", pred.shape, pred.dtype)
        print("mask_tum shape and type:", mask_tum.shape, mask_tum.dtype)
        
        metrics = compute_volume_metrics(
            y_pred=pred,
            y_true=mask_tum,
            device=torch.device(
                "cuda"
                if torch.cuda.is_available()
                and np.prod(img_sitk.GetSize()) < threshold_img_size
                else "cpu"
            ),
        )
        for key in metrics.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df.loc[index, key + "_phd"] = metrics[key]
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df.loc[index, "std_phd"] = float(preds.std())
                df.loc[index, "interdice_phd"] = np.mean(interdices)


def write_preds_in_db(preds_paths_list, RADIOMICS, IAM, REFRESH_TOKEN, ARGO, args):
    """
    Write a list of nrrd files in SOPHiA .db and convert them to meshes to be seen in DDM.
    """

    Requester = create_requester(RADIOMICS, IAM, REFRESH_TOKEN)

    history_path_relative = os.path.join(
        "phd_robustness", args.model_version, "History"
    )
    history_path = os.path.join(args.output_path, args.model_version, "History")
    os.makedirs(history_path, exist_ok=True)

    request_path = os.path.join(args.output_path, args.model_version, "Request")
    os.makedirs(request_path, exist_ok=True)

    manage_segmentation_conversion(
        preds_paths_list,
        history_path_relative,
        REFRESH_TOKEN,
        len(preds_paths_list),
        datetime.now(timezone("Europe/Paris")).strftime("%d_%m_%Y__%H_%M_%S"),
        request_path,
        ARGO,
        Requester,
        write_in_db=True,
    )


def compute_volume_metrics(
    y_pred: torch.tensor,
    y_true: torch.tensor,
    device: torch.device,
    voxel_volume: float | None = None,
) -> dict[str, float]:
    """
    Return a dict of computed metrics comparing predictions and ground truth.
    """
    if y_pred.size() != y_true.size():
        msg = f"Prediction size (got {y_pred.size()}) must match Ground truth size (got {y_true.size()})."
        raise ValueError(msg)

    y_pred = y_pred.flatten().to(device)
    y_true = y_true.flatten().to(device)

    nb_voxel_gt = torch.sum(y_true).item()
    if nb_voxel_gt == 0:
        return ValueError("Ground truth can not be empty.")

    scores = {}
    nb_voxel_pred = torch.sum(y_pred).item()
    scores["pred_size_nb_voxel"] = nb_voxel_pred
    scores["gt_size_nb_voxel"] = nb_voxel_gt
    if voxel_volume is not None:
        scores["pred_size_mm3"] = voxel_volume * nb_voxel_pred
        scores["gt_size_mm3"] = voxel_volume * nb_voxel_gt
    dice, iou = Dice(num_classes=1, multiclass=False).to(
        device
    ), BinaryJaccardIndex().to(device)
    try:
        scores["dice"] = dice(y_pred, y_true).item()
        #scores["iou"] = iou(y_pred, y_true).item()
    except Exception as e:
        print(e)
        scores["dice"]=np.nan
        scores["iou"]=np.nan
    return scores


def copy_conversion_info(
    sitk_image: sitk.Image,
    series_uid: str,
    model_name_version: str,
    suffix: str,
) -> None:
    """
    Add metadata to the segmentation to help conversion.
    """
    sitk_image.SetMetaData(
        "conversion",
        json.dumps(
            {
                "tum": {
                    "labels": [1],
                    "postprocessing": 0,
                    "organ_id": 40,
                    "compartment_id": 1,
                    "target": 1,
                    "lesion_name": "tum_phd",
                    "histology": 0,
                    "model_name_version": model_name_version + "_" + suffix,
                },
                "serie_uid": series_uid,
            }
        ),
    )