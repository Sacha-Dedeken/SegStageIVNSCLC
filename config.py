from str2bool import str2bool
from argparse import ArgumentParser


def initialize_parser():
    """
    Function that add to the parser a list of arguments with typing and help.

    Returns:
        parser (ArgumentParser) : the parser that will be used for this run.
    """

    parser = ArgumentParser()
    
    parser.add_argument(
        "--train_size",
        default=250,
        type=int,
        help="Number of train images used in train set.",
    )

    parser.add_argument(
        "--unique_input",
        type=str2bool,
        help="Do you give a single manifest that need to be split or do you use 2/3 csv (if so, specify in manifest.py) ?",
    )

    parser.add_argument(
        "--tumor_only",
        type=str2bool,
        help="Do you want to train the model to predict only tumor mask (or also Lung, if so put False)",
    )

    parser.add_argument(
        "--job_mode",
        default="train",
        type=str,
        choices=["train", "evaluate", "deploy", "convert", "evaluate_from_dicom"],
        help="Determine if you want to train+test, test, train or just save in db",
    )

    parser.add_argument(
        "--save_pred",
        type=str2bool,
        help="Do you want to save test predictions in nrrd files on storage?",
    )

    parser.add_argument(
        "--mask2mesh",
        type=str2bool,
        help="Do you want to convert saved nrrd preds into meshes viewable in db and ddm?",
    )

    parser.add_argument(
        "--preds2convert_path",
        default="Inputs/conversion_paths.json",
        type=str,
        help="If job_mode=='convert', use this list of paths to run conversion in meshes.",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        help="model name and version used in nrrd metadata and in .db",
    )

    parser.add_argument(
        "--manifest_path",
        default="data/bigtum_manifest.csv",
        type=str,
        help="If unique_input, will split this manifest for the training.",
    )

    parser.add_argument(
        "--exp_folder",
        type=str,
        help="AML path to the model folder.",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="suffix for this experiment files",
    )

    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="This run will use this fold index as the test fold of the cross-validation (used for parallel execution).",
    )

    parser.add_argument(
        "--image_dataset",
        type=str,
        help="AML path to the data folder for this run.",
    )

    parser.add_argument(
        "--dcm_dir",
        type=str,
        help="AML path to the DICOM folder.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="AML path to the folder to load current_state_dict.",
    )

    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="The training will stop if it reached this number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="All the dataloaders will use this batch size.",
    )
    parser.add_argument(
        "--image_size_z",
        default=128,
        type=int,
        help="Images will be resampled to this size on Z axis.",
    )
    parser.add_argument(
        "--image_size_x",
        default=128,
        type=int,
        help="Images will be resampled to this size on X axis.",
    )
    parser.add_argument(
        "--image_size_y",
        default=128,
        type=int,
        help="Images will be resampled to this size on Y axis.",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="The training will start with this learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="The weight_decay (L2 regularization) used during training",
    )

    parser.add_argument(
        "--norm_type",
        default="INSTANCE",
        help="The normalization type (None, BATCH, INSTANCE) used during training",
    )

    parser.add_argument(
        "--accumulation_steps",
        default=1,
        type=int,
        help="The number of accumulated gradients batches during training.",
    )

    parser.add_argument(
        "--lr_patience",
        default=1,
        type=int,
        help="Number of epochs with no improvement after which learning rate will be reduced.",
    )

    parser.add_argument(
        "--lr_cooldown",
        default=1,
        type=int,
        help="Number of epochs to wait before resuming normal operation after lr has been reduced.",
    )

    parser.add_argument(
        "--lr_factor",
        default=0.99,
        type=float,
        help="Factor by which the learning rate will be reduced.",
    )

    parser.add_argument(
        "--early_stopping",
        default=40,
        type=int,
        help="Number of epochs with no improvement after which training phase will be stopped.",
    )

    parser.add_argument(
        "--only_test",
        type=str2bool,
        help="If True, don't train the model and directly test it on the whole dataset.",
    )
    parser.add_argument(
        "--params_to_test",
        default="data/best-model-parameters.pt",
        type=str,
        help="If only_test=True, select the weights at this path to initialize the model.",
    )

    parser.add_argument(
        "--cprofile",
        type=str2bool,
        help="Path to the data manifest (one patient per row, columns contain target, images paths, possibly clinics and other variables).",
    )

    parser.add_argument(
        "--test_size",
        default=1 / 6,
        type=float,
        help="Proportion of data that will be used for final test.",
    )

    parser.add_argument(
        "--clip",
        type=str,
        help="Choose clip between 'no', 'lung', 'abdo'.",
    )

    parser.add_argument(
        "--scheduler_strategy",
        default="ReduceLROnPlateau",
        help="Define the LRScheduler used during training.",
    )

    parser.add_argument(
        "--strategy_metric",
        default="mean_tum_dice",
        help="Choose the validation metric to use for early_stopping and selecting the best model weights.",
    )

    parser.add_argument(
        "--strategy_mode",
        default="max",
        help="Choose if the strategy_metric has to be maximize or minimize.",
    )

    parser.add_argument(
        "--random_seed",
        default=2024,
        type=int,
        help="This random seed will be used by random, numpy and pytorch during the script.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="name to select a model configuration.",
    )

    parser.add_argument(
        "--DA",
        default="baseline",
        type=str,
        help="select a data augmentation function",
    )
    
    parser.add_argument(
        "--forward_passes_MCDO",
        default=1,
        type=int,
        help="Number of forward passes to do at inference with Monte Carlo Dropout",
    )

    return parser
