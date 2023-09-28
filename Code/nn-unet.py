# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import shutil
import sys

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.handlers.early_stopping import EarlyStopping
from ignite.engine.events import Events

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
                #! NEW # transform all non 0 labels to 1
                AsDiscreted(keys[1], threshold=0.5)
            ]
        )
        dtype = (torch.float32, torch.uint8)
    if mode == "val":
        #! NEW # transform all non 0 labels to 1
        xforms.extend([AsDiscreted(keys[1], threshold=0.5)])
        dtype = (torch.float32, torch.uint8)
    if mode == "infer":
        dtype = (torch.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    
    return monai.transforms.Compose(xforms)


def get_kernels_strides():
	sizes, spacings = [192, 192, 16], [0.764318645, 0.764318645, 5.238947283]
	strides, kernels = [], []

	while True:
		spacing_ratio = [sp / min(spacings) for sp in spacings]
		stride = [
			2 if ratio <= 2 and size >= 8 else 1
			for (ratio, size) in zip(spacing_ratio, sizes)
		]
		kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
		if all(s == 1 for s in stride):
			break
		sizes = [i / j for i, j in zip(sizes, stride)]
		spacings = [i * j for i, j in zip(spacings, stride)]
		kernels.append(kernel)
		strides.append(stride)
	strides.insert(0, len(spacings) * [1])
	kernels.append(len(spacings) * [3])
	return kernels, strides

def get_net():
	kernels, strides = get_kernels_strides()

	net = monai.networks.nets.DynUNet(
		spatial_dims = 3,
		in_channels = 1,
		out_channels = 2,
		kernel_size = kernels,
		strides = strides,
		upsample_kernel_size = strides[1:],
		norm_name = "instance",
	)

	return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


def train(train_folder="data/train", val_folder="data/val", model_folder="runs/nn-unet"):
    """run a training pipeline."""

    train_image_path = os.path.join(train_folder, "ribfrac-train-images")
    train_label_path = os.path.join(train_folder, "ribfrac-train-labels")
    val_image_path = os.path.join(val_folder, "ribfrac-val-images")
    val_label_path = os.path.join(val_folder, "ribfrac-val-labels")
    
    train_images = sorted(glob.glob(os.path.join(train_image_path, "*-image.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_label_path, "*-label.nii.gz")))
    
    val_images = sorted(glob.glob(os.path.join(val_image_path, "*-image.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(val_label_path, "*-label.nii.gz")))

    logging.info(f"training: image/label ({len(train_images)}) folder: {train_folder}")
    logging.info(f"validation: image/label ({len(val_images)}) folder: {val_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(train_images, train_labels)]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(val_images, val_labels)]

    # create a training data loader
    batch_size = 2
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = get_net().to(device)
    max_epochs, lr, momentum = 500, 1e-4, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2)]
    )
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        postprocessing=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True), name="StatsHandler"),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )

    #! NEW EARLY STOPPING REGULARIZATION
    early_stopper = EarlyStopping(patience = 30, score_function = get_dice_score,
	                              trainer = trainer, min_delta = 0.01, cumulative_delta = True)
    evaluator.add_event_handler(event_name = Events.EPOCH_COMPLETED, handler = early_stopper)
    print("START TRAINING")
    trainer.run()

def get_dice_score(engine):
	score = engine.state.metrics
	print("THE MEAN DICE: ", score["val_mean_dice"])
	return score["val_mean_dice"]

def infer(data_folder="data/test", model_folder="runs/nn-unet", prediction_folder="output/nn-unet"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    data_folder = os.path.join(data_folder, "ribfrac-test-images")
    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*-image.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.transforms.SaveImage(output_dir=prediction_folder, mode="nearest", resample=True, separate_folder=False)
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image'].meta['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            for p in preds:  # save each image+metadata in the batch respectively
                saver(p)

    files = os.listdir(prediction_folder)
    for f in files:
        new_name = f.replace("image_trans", "label")
        os.rename(os.path.join(prediction_folder, f), os.path.join(prediction_folder, new_name))



if __name__ == "__main__":
    """
    Usage:
        python Code/nn-unet.py train --train_folder "data/train" --val_folder "data/val" # run the training pipeline
        python Code/nn-unet.py infer --test_folder "data/test" # run the inference pipeline
    """
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--train_folder", default="data/train", type=str, help="training data folder")
    parser.add_argument("--val_folder", default="data/val", type=str, help="validation data folder")
    parser.add_argument("--test_folder", default="data/test", type=str, help="test data folder")
    parser.add_argument("--model_folder", default="runs/nn-unet", type=str, help="model folder")
    args = parser.parse_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        train_folder = args.train_folder
        val_folder = args.val_folder
        train(train_folder=train_folder, val_folder=val_folder, model_folder=args.model_folder)
    elif args.mode == "infer":
        test_folder = args.test_folder
        infer(data_folder=test_folder, model_folder=args.model_folder)
    else:
        raise ValueError("Unknown mode.")