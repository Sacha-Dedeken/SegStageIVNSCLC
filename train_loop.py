import torch
import time
import pickle
import numpy as np
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import colors

from torchmetrics.classification import Dice

from model import AttentionUnet
from monai.networks.nets import UNet, UNETR, SwinUNETR
from utils import (
    EarlyStopping,
    count_parameters,
    save_random_state_dict,
    load_random_state_dict,
)
from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss

import cProfile

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dice = Dice(num_classes=1, multiclass=False).to(device)


def train_loop(
    train_dataloader,
    val_dataloader,
    args,
):
    """
    Train the model over train/validation dataloaders and save metrics and parameters.
    """

    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.tumor_only:
        sample_metrics_names = ["patient_id", "tum_loss", "loss", "tum_dice"]
        epoch_metrics_names = [
            "mean_tum_loss",
            "mean_loss",
            "mean_tum_dice",
            "Q1_tum_dice",
            "Q2_tum_dice",
            "Q3_tum_dice",
        ]

    else:
        sample_metrics_names = [
            "patient_id",
            "lung_loss",
            "tum_loss",
            "loss",
            "lung_dice",
            "tum_dice",
        ]
        epoch_metrics_names = [
            "mean_lung_loss",
            "mean_tum_loss",
            "mean_loss",
            "mean_lung_dice",
            "Q1_lung_dice",
            "Q2_lung_dice",
            "Q3_lung_dice",
            "mean_tum_dice",
            "Q1_tum_dice",
            "Q2_tum_dice",
            "Q3_tum_dice",
        ]

    sample_metrics = {
        m: {"train": [], "val": []} for m in ["patient_id"] + sample_metrics_names
    }
    epoch_metrics = {m: {"train": [], "val": []} for m in epoch_metrics_names}

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

    print(
        f"{type(model).__name__} has {count_parameters(model)} trainable parameters.",
        flush=True,
    )

    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.detach().cpu().numpy().tolist()

    # Save the parameters to a JSON file
    json_filename = "outputs/model_parameters_ep0.json"
    with open(json_filename, "w") as json_file:
        json.dump(params_dict, json_file)

    model = model.to(device)

    Loss = DiceFocalLoss(sigmoid=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_patience, gamma=args.lr_factor
    )

    early_stopping = EarlyStopping(
        tolerance=args.early_stopping, mode=args.strategy_mode
    )

    epoch = 0

    print("\nSTART OF THE TRAINING PHASE\n---------------------", flush=True)

    best_epoch, best_metric, best_params = -1, 100, deepcopy(model.state_dict())
    torch.save(best_params, f"outputs/best-model-parameters_{args.suffix}.pt")

    checkpoint_path = os.path.join(
        args.output_path,
        args.model_version,
        "Checkpoints/",
        f"checkpoint_{args.suffix}_{args.fold}.pt",
    )

    print(f"--check if a .pt file exists at path: {checkpoint_path}", flush=True)
    if os.path.isfile(checkpoint_path):
        restart = True
        print("found an existing file", flush=True)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"))

        model.load_state_dict(checkpoint["weights"])
        start_ep = checkpoint["epoch"] + 1
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=checkpoint["lr"],
            weight_decay=args.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        early_stopping.counter = checkpoint["early_stopping_counter"]
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_patience, gamma=args.lr_factor
        )
        epoch_metrics = checkpoint["epoch_metrics"]
        best_epoch = checkpoint["best_epoch"]

        load_random_state_dict(checkpoint["random_state_dict"])

    else:
        restart = False
        print("didn't found an existing file, restart from epoch 0", flush=True)
        start_ep = 0
        epoch_metrics = {m: {"train": [], "val": []} for m in epoch_metrics_names}

    for epoch in range(start_ep, args.max_epochs):
        print(f"\n--- EPOCH {epoch}", flush=True)

        if epoch != start_ep:
            print(f"Epoch {epoch-1} duration: {time.time() - clock}", flush=True)
        clock = time.time()

        if epoch == start_ep + 3 and args.cprofile:
            profiler.disable()
            profiler.dump_stats("outputs/profiler_results.prof")

        sample_metrics = {
            m: {"train": [], "val": []} for m in ["patient_id"] + sample_metrics_names
        }

        # (TRAIN DATALOADER)

        sample_metrics = step(
            model,
            train_dataloader,
            epoch,
            "val",
            Loss,
            sample_metrics,
            optimizer,
            args,
            restart,
        )
        restart = False

        update_metrics(epoch_metrics, sample_metrics, phase="train")

        # (VAL DATALOADER)

        sample_metrics = step(
            model,
            val_dataloader,
            epoch,
            "val",
            Loss,
            sample_metrics,
            optimizer,
            args,
            restart=False,
        )

        update_metrics(epoch_metrics, sample_metrics, phase="val")

        update_plots(epoch_metrics)

        with open("outputs/metrics_dict.pkl", "wb") as metrics_file:
            pickle.dump(epoch_metrics, metrics_file)

        metrics_path = os.path.join(
            args.output_path,
            args.model_version,
            "grid_search",
            args.suffix,
            "fold_" + str(args.fold),
        )
        os.makedirs(metrics_path, exist_ok=True)
        if epoch == 0:
            print(metrics_path)
        with open(os.path.join(metrics_path, "metrics_dict.pkl"), "wb") as metrics_file:
            pickle.dump(epoch_metrics, metrics_file)

        scheduler.step()
        print(f"\nLR epoch {epoch}: {optimizer.param_groups[0]['lr']} ", flush=True)

        early_stopping(epoch_metrics[args.strategy_metric]["val"][-1])

        if early_stopping.counter == 0:
            best_epoch, best_metric, best_params = (
                epoch,
                epoch_metrics[args.strategy_metric]["val"][-1],
                deepcopy(model.state_dict()),
            )
            torch.save(best_params, f"outputs/best-model-parameters_{args.suffix}.pt")
            print(
                f"\n--> EP{best_epoch} with metric {args.strategy_metric}={best_metric} is the new best model !",
                flush=True,
            )

        checkpoint = {
            "epoch": epoch,
            "optimizer": deepcopy(optimizer.state_dict()),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_metrics": epoch_metrics,
            "weights": deepcopy(model.state_dict()),
            "early_stopping_counter": early_stopping.counter,
            "best_epoch": best_epoch,
            "best_params": best_params,
            "random_state_dict": save_random_state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        if early_stopping.early_stop:
            print("EARLY STOPPING ACTIVATED We are at epoch:", epoch, flush=True)
            break
        else:
            print(
                "EARLY STOPPING COUNTER:",
                early_stopping.counter,
                "/",
                early_stopping.tolerance,
                "\n",
                flush=True,
            )

    return model, checkpoint


def step(
    model, dataloader, epoch, phase, loss_func, metrics, optimizer, args, restart=False
):
    """ """

    if phase == "train":
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    with grad_context:
        for i, data in enumerate(dataloader):
            if args.tumor_only:
                img, mask_tum, patient_id = data
            else:
                img, mask_tum, patient_id, mask_lung = data
                mask_lung = mask_lung.to(device)

            img = img.as_tensor()
            img = img.to(device)
            mask_tum = mask_tum.to(device)

            output = model(img)

            if args.tumor_only:
                tum_loss = loss_func(output[:, 0], mask_tum)
                combined_loss = tum_loss

            else:
                tum_loss = loss_func(output[:, 0], mask_tum)
                lung_loss = loss_func(output[:, 1], mask_lung)
                combined_loss = lung_loss + tum_loss

            if phase == "train":
                combined_loss /= args.accumulation_steps
                combined_loss.backward()
                if (i + 1) % args.accumulation_steps == 0 or i == len(dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()

            for j, s in enumerate(output):
                metrics["patient_id"][phase].append(patient_id[j])
                metrics["tum_loss"][phase].append(loss_func(s[0], mask_tum[j]).item())
                metrics["tum_dice"][phase].append(compute_dice(s[0], mask_tum[j]))
                if not args.tumor_only:
                    metrics["lung_loss"][phase].append(
                        loss_func(s[1], mask_lung[j]).item()
                    )
                    metrics["loss"][phase].append(
                        metrics["lung_loss"][phase][-1] + metrics["tum_loss"][phase][-1]
                    )
                    metrics["lung_dice"][phase].append(compute_dice(s[1], mask_lung[j]))
                else:
                    metrics["loss"][phase].append(metrics["tum_loss"][phase][-1])

    if epoch in [0, 2, 8, 16, 24, 32, 40, 64, 96, 128, 160, 192] or restart:
        if args.tumor_only:
            plot_imshow(epoch, img, output, mask_tum, patient_id)
        else:
            plot_imshow(epoch, img, output, mask_tum, patient_id, mask_lung)

    return metrics


def compute_dice(output_sample, mask_sample):
    """ """

    return dice((torch.sigmoid(output_sample) > 0.5) * 1, mask_sample.int()).item()


def update_plots(epoch_metrics):
    """ """

    for m in epoch_metrics.keys():
        if m[:4] == "mean":
            plt.figure(m)
            plt.clf()
            plt.plot(epoch_metrics[m]["train"], c="navy", label="train")
            plt.plot(epoch_metrics[m]["val"], c="darkorange", label="val")
            plt.title(f"{m} over epoches")
            plt.legend()
            plt.savefig(f"outputs/{m}.png")

    for phase in ["train", "val"]:
        plt.figure(phase + "_tum_dice_distrib")
        plt.clf()
        plt.plot(
            epoch_metrics["Q2_" + "tum_dice"][phase],
            c="darkorange",
            alpha=1,
            label="Q2",
        )
        plt.fill_between(
            list(range(len(epoch_metrics["Q2_tum_dice"][phase]))),
            epoch_metrics["Q3_tum_dice"][phase],
            epoch_metrics["Q1_tum_dice"][phase],
            alpha=0.2,
            color="orange",
        )
        plt.title(phase + "_tum_dice_distrib")
        plt.savefig("outputs/" + phase + "_tum_dice_distrib.png")


def plot_imshow(epoch, img, output, mask_tum, patient_id, mask_lung=False):
    """ """

    img_plot = img.detach().cpu().numpy()
    output_plot = output.detach().cpu().numpy()

    mask_tum_plot = mask_tum.detach().cpu().numpy()
    combined_mask = np.zeros_like(mask_tum_plot, dtype=np.uint8)

    if torch.is_tensor(mask_lung):
        mask_lung_plot = mask_lung.detach().cpu().numpy()
    else:
        mask_lung_plot = np.zeros_like(mask_tum_plot)
        combined_mask[mask_lung_plot == 1] = 1  # Lung regions (blue)

    combined_mask[mask_tum_plot == 1] = 2  # Tumor regions (red)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), num=str(patient_id[0]))

    axs[0, 0].imshow(img_plot[0, 0, img_plot.shape[2] // 2], cmap="gray")
    axs[0, 1].imshow(output_plot[0, 0, output_plot.shape[2] // 2], cmap="gray")
    if torch.is_tensor(mask_lung):
        axs[1, 0].imshow(output_plot[0, 1, output_plot.shape[2] // 2], cmap="gray")
    else:
        axs[1, 0].imshow(output_plot[0, 0, output_plot.shape[2] // 2], cmap="gray")
    axs[1, 1].imshow(
        combined_mask[0, combined_mask.shape[2] // 2],
        cmap=colors.ListedColormap(["black", "blue", "red"]),
    )

    plt.suptitle(f"{patient_id[0]}")

    plt.savefig(f"outputs/ep_{epoch}_{patient_id[0]}.png")


def update_metrics(epoch_metrics, sample_metrics, phase):
    """ """

    print(f"\n[[{phase.upper()}]]")

    for m in epoch_metrics.keys():
        if m[:4] == "mean":
            epoch_metrics[m][phase].append(np.mean(sample_metrics[m[5:]][phase]))
        elif m[:2] == "Q1":
            epoch_metrics[m][phase].append(
                np.quantile(sample_metrics[m[3:]][phase], 0.25)
            )
        elif m[:2] == "Q2":
            epoch_metrics[m][phase].append(
                np.quantile(sample_metrics[m[3:]][phase], 0.5)
            )
        elif m[:2] == "Q3":
            epoch_metrics[m][phase].append(
                np.quantile(sample_metrics[m[3:]][phase], 0.75)
            )

        print(f"{m}={epoch_metrics[m][phase]}", flush=True)

    return epoch_metrics
