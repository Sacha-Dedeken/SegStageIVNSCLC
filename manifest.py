import pandas as pd
import torch
import os
import multiprocessing
from sklearn.model_selection import train_test_split, KFold

from dataset import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manifest_reader(args):
    """
    Customizable function that takes as input the job arguments and return Pytorch dataloaders.

    Args:
        args (Namespace): job arguments from the parser
    """

    if args.unique_input:
        if args.job_mode in ["train", "evaluate"]:
            train_manifest, val_manifest, test_manifest = train_val_test_kfold_split(
                df=pd.read_csv(args.manifest_path),
                n_splits=5,
                fold=args.fold,
                random_seed=args.random_seed,
            )

        elif args.job_mode == "deploy":
            train_manifest, val_manifest = train_test_split(
                pd.read_csv(args.manifest_path),
                test_size=args.test_size,
                random_state=args.random_seed,
            )
            test_manifest = list()

    else:
        # customizable part depending of the data sources

        train_manifest = pd.read_csv(
            f"data/cross_val_splits/fold_{args.fold}/train.csv"
        )[:args.train_size]
        val_manifest = pd.read_csv(f"data/cross_val_splits/fold_{args.fold}/val.csv")
        test_manifest = pd.read_csv(f"data/cross_val_splits/fold_{args.fold}/test.csv")

        part_to_remove = len(
            "/mnt/azureml/cr/j/97be1bdb89024b84811abf34a8b258d1/cap/data-capability/wd/container_dir/ImageNrrd/"
        )
        
        manifests = {
            "train": train_manifest,
            "val": val_manifest,
            "test": test_manifest,
        }

        for key in manifests.keys():
            manifests[key]["img_path"] = manifests[key]["img_path"].apply(
                lambda x: os.path.join(args.image_dataset, x[part_to_remove:])
            )
            manifests[key]["mask_tum"] = manifests[key]["mask_tum"].apply(
                lambda x: os.path.join(args.image_dataset, x[part_to_remove:])
            )

            if not args.tumor_only:
                manifests[key]["mask_lung"] = manifests[key]["mask_lung"].apply(
                    lambda x: os.path.join(args.image_dataset, x[part_to_remove:])
                )

            manifests[key].drop_duplicates(subset=["patient_id"], inplace=True)
            manifests[key].to_csv(f"outputs/{key}_manifest.csv", index=False)

            print(f"-size of {key}_manifest: {len(manifests[key])}")

    if args.job_mode in ["train", "evaluate", "deploy"]:
        nb_worker = multiprocessing.cpu_count()
        print("nb_worker used: ", nb_worker)

        train_dataset = MyDataset(
            dataset=train_manifest,
            args=args,
            DA=args.DA,
            device=device,
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=nb_worker,
            pin_memory=True,
        )

        val_dataset = MyDataset(
            dataset=val_manifest,
            args=args,
            DA="no_DA",
            device=device,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=nb_worker,
            pin_memory=True,
        )

        return train_dataloader, val_dataloader, test_manifest

    else:
        return None, None, test_manifest


def train_val_test_kfold_split(df, n_splits, fold, random_seed=90):
    """
    Realize a K-fold cross-validation and return the fold split.

    Args:
        df (pd.DataFrame): the dataframe to be split
        n_splits: the number of folds
        fold: the fold to use as test set
        random_seed: used to execute the random split
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    train_val_idx, test_idx = [i for i in kf.split(df)][fold]

    train_val_manifest = df.loc[train_val_idx]
    test_manifest = df.loc[test_idx]

    train_manifest, val_manifest = train_test_split(
        train_val_manifest,
        test_size=1 / (n_splits - 1),
        random_state=random_seed,
    )

    print("train indexes: ", train_manifest.index.tolist())
    print("val indexes: ", val_manifest.index.tolist())
    print("test indexes: ", test_manifest.index.tolist())

    return train_manifest, val_manifest, test_manifest
