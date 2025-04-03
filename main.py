import os

import torch
import json
from monai.utils import set_determinism

from config import initialize_parser
from manifest import manifest_reader
from train_loop import train_loop
from evaluate import evaluate_model, write_preds_in_db
from model import AttentionUnet
from monai.networks.nets import UNet, UNETR, SwinUNETR

print("\n---\nSTART OF THE SCRIPT\n---\n", flush=True)

parser = initialize_parser()

args = parser.parse_args()

for arg in vars(args):
    print(f"-{arg}: {getattr(args, arg)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device, flush=True)

# Set random seeds

set_determinism(seed=args.random_seed)

# create dataloaders from manifest

train_dataloader, val_dataloader, test_manifest = manifest_reader(args)

# ----------- TRAINING AND EVALUATION

if args.job_mode in ["train", "deploy"]:
    model, current_state_dict = train_loop(
        train_dataloader,
        val_dataloader,
        args,
    )

    print(f"load the best_params of the best_epoch: {current_state_dict['best_epoch']}")
    model.load_state_dict(current_state_dict["best_params"])

if args.job_mode in ["evaluate", "evaluate_from_dicom"]:
    if args.model_name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
            norm_type=args.norm_type,
        )
    elif args.model_name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
            norm=args.norm_type,
        )
    elif args.model_name == "unetr":
        model = UNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            img_size=(128, 128, 128),
            dropout_rate=0.2,
        )
    elif args.model_name == "swin_unetr":
        model = SwinUNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            img_size=(128, 128, 128),
            drop_rate=0.2,
        )

    if args.params_to_test[: len("Checkpoints")] == "Checkpoints":
        current_state_dict = torch.load(
            os.path.join(args.exp_folder, args.model_version, args.params_to_test),
            map_location=torch.device("cuda"),
        )
        model.load_state_dict(current_state_dict["best_params"])

    else:
        model.load_state_dict(
            torch.load(
                os.path.join(args.exp_folder, args.model_version, args.params_to_test)
            )
        )

if args.job_mode == "convert":
    with open(
        os.path.join(args.exp_folder, args.model_version, args.preds2convert_path), "r"
    ) as json_file:
        preds_paths_list = json.load(json_file)
        if len(preds_paths_list) > 0:
            print(
                f"convert mode, json loaded, ready to convert {len(preds_paths_list)} files of the form {preds_paths_list[0]}"
            )
        else:
            print("convert mode, json loaded, no files to convert.")
elif args.job_mode in ["train", "evaluate", "evaluate_from_dicom"]:
    preds_paths_list = evaluate_model(model, test_manifest, args)
else:
    preds_paths_list = list()

if len(preds_paths_list) > 0:
    print(
        f"{args.job_mode} mode, evaluation ended, ready to convert {len(preds_paths_list)} files of the form {preds_paths_list[0]}"
    )
else:
    print(f"{args.job_mode} mode, json loaded, no files to convert.")

if len(preds_paths_list) > 0:
    with open("env.json", "r") as file:
        env = json.load(file)

    if args.mask2mesh:
        write_preds_in_db(
            preds_paths_list,
            env["RADIOMICS"],
            env["IAM"],
            env["REFRESH_TOKEN"],
            env["ARGO"],
            args,
        )

print("\n----\n END \n----", flush=True)
