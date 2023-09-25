# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#THIS IS OLD CODE FROM BOGDAN's PROJECT. IT WORKED FOR COVID DATA.

import argparse
import logging
import os
import shutil
import sys

import glob
import monai
import numpy as np
import sklearn
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine.events import Events
from ignite.handlers.early_stopping import EarlyStopping
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, \
	TensorBoardStatsHandler, TensorBoardImageHandler
from monai.transforms import (
	AddChanneld,
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
	ToTensord,
	Activationsd,
)

print('sklearn version:')
print(sklearn.__version__)

print('sklearn version:')
print(sklearn.__version__)


def get_xforms(mode = "train", keys = ("image", "label")):
	"""returns a composed transform for train/val/infer."""
	print('start get_xforms function')
	xforms = [
		LoadImaged(keys),
		AddChanneld(keys),
		Orientationd(keys, axcodes = "LPS"),
		Spacingd(keys, pixdim = (0.764318645, 0.764318645, 5.238947283), mode = ("bilinear", "nearest")[: len(keys)]),
		ScaleIntensityRanged(keys[0], a_min = -1050.0, a_max = -150.0, b_min = 0.0, b_max = 1.0, clip = True),
	]
	if mode == "train":
		xforms.extend(
			[
				SpatialPadd(keys, spatial_size = (192, 192, -1),
				            mode = "reflect"),  # ensure at least 192x192
				RandAffined(
					keys,
					prob = 0.15,
					# 3 parameters control the transform on 3 dimensions
					rotate_range = (0.05, 0.05, None),
					scale_range = (0.1, 0.1, None),
					mode = ("bilinear", "nearest"),
					as_tensor_output = False,
				),
				RandCropByPosNegLabeld(keys, label_key = keys[1], spatial_size = (256, 256, 16), num_samples = 3),

				# Set prob to 0 to not add noise
				RandGaussianNoised(keys[0], prob = 0.15, std = 0.01),

				RandFlipd(keys, spatial_axis = 0, prob = 0.5),
				RandFlipd(keys, spatial_axis = 1, prob = 0.5),
				RandFlipd(keys, spatial_axis = 2, prob = 0.5),
			]
		)
		print(keys)
		dtype = (np.float32, np.uint8)
	if mode == "val":
		dtype = (np.float32, np.uint8)
	if mode == "infer":
		dtype = (np.float32,)
	xforms.extend([CastToTyped(keys, dtype = dtype), ToTensord(keys)])
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


def get_inferer(_mode = None):
	"""returns a sliding window inference instance."""
	print('start get_inferer function')
	patch_size = (192, 192, 16)
	sw_batch_size, overlap = 8, 0.5
	inferer = monai.inferers.SlidingWindowInferer(
		roi_size = patch_size,
		sw_batch_size = sw_batch_size,
		overlap = overlap,
		mode = "constant",
		padding_mode = "replicate",
	)
	return inferer


class DiceCELoss(nn.Module):
	"""Dice and Xentropy loss"""

	def __init__(self):
		super().__init__()
		print('start init_loss function')
		self.cross_entropy = nn.CrossEntropyLoss()
		self.tversky = monai.losses.TverskyLoss(to_onehot_y = True, softmax = True, alpha = 0.7, beta = 0.3)

	def forward(self, y_pred, y_true):
		cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim = 1).long())
		tverskylos = self.tversky(y_pred, y_true)
		return tverskylos + cross_entropy


def reset_weights(m):
	"""
	  Reset model weights to avoid
	  weight leakage.
	"""
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()


def get_dice_score(engine):
	score = engine.state.metrics
	print("THE MEAN DICE: ", score["val_mean_dice"])
	return score["val_mean_dice"]


def train(data_folder = ".", model_folder = "."):
	"""run a training pipeline."""

	print('start train function')
	images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
	labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
	logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

	amp = True
	keys = ("image", "label")
	train_frac, val_frac = 0.8, 0.2
	n_train = int(train_frac * len(images)) + 1
	n_val = min(len(images) - n_train, int(val_frac * len(images)))
	logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

	print('--------------------------------')
	train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
	val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
	batch_size = 2
	logging.info(f"batch size {batch_size}")
	train_transforms = get_xforms("train", keys)

	train_ds = monai.data.CacheDataset(data = train_files, transform = train_transforms)
	train_loader = monai.data.DataLoader(
		train_ds,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 2,
		pin_memory = torch.cuda.is_available(),
	)

	# create a validation data loader
	val_transforms = get_xforms("val", keys)
	val_ds = monai.data.CacheDataset(data = val_files, transform = val_transforms)

	val_loader = monai.data.DataLoader(
		val_ds,
		batch_size = 1,  # image-level batch to the sliding window method, not the window-level batch
		num_workers = 2,
		pin_memory = torch.cuda.is_available(),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = get_net().to(device)
	net.apply(reset_weights)
	max_epochs, lr = 1000, 1e-3
	logging.info(f"epochs {max_epochs}, lr {lr}")
	opt = torch.optim.Adam(net.parameters(), lr = lr)

	# create evaluator (to be used to measure model quality during training)
	val_post_transform = monai.transforms.Compose(
		[
			Activationsd(keys = "pred", softmax = True),
			AsDiscreted(keys = ("pred", "label"), argmax = (True, False),
			            to_onehot = True, n_classes = 2, threshold_values = True),
		]
	)
	val_handlers = [
		ProgressBar(),
		StatsHandler(output_transform = lambda x: None),
		TensorBoardStatsHandler(log_dir = model_folder, output_transform = lambda x: None),
		TensorBoardImageHandler(
			log_dir = model_folder,
			batch_transform = lambda x: (x["image"], x["label"]),
			output_transform = lambda x: x["pred"],
		),
		CheckpointSaver(save_dir = model_folder, save_dict = {"net": net}, save_key_metric = True, key_metric_n_saved = 1),
	]
	evaluator = monai.engines.SupervisedEvaluator(
		device = device,
		val_data_loader = val_loader,
		network = net,
		inferer = get_inferer(),
		post_transform = val_post_transform,
		key_val_metric = {

			"val_mean_dice": MeanDice(include_background = False,
			                          output_transform = lambda x: (x["pred"], x["label"])),
		},
		val_handlers = val_handlers,
		amp = amp,
	)

	train_post_transforms = monai.transforms.Compose(
		[
			Activationsd(keys = "pred", softmax = True),
			AsDiscreted(keys = "pred", threshold_values = True),
		]
	)

	train_handlers = [
		ValidationHandler(validator = evaluator, interval = 1, epoch_level = True),
		StatsHandler(tag_name = "train_loss", output_transform = lambda x: x["loss"]),
		TensorBoardStatsHandler(log_dir = model_folder, tag_name = "train_loss", output_transform = lambda x: x["loss"]),
	]

	trainer = monai.engines.SupervisedTrainer(
		device = device,
		max_epochs = max_epochs,
		train_data_loader = train_loader,
		network = net,
		optimizer = opt,
		loss_function = DiceCELoss(),
		inferer = get_inferer(),
		post_transform = train_post_transforms,
		key_train_metric = {"train_mean_dice": MeanDice(
			include_background = False, output_transform = lambda x: (x["pred"], x["label"]))},
		train_handlers = train_handlers,
		amp = amp,
	)

	early_stopper = EarlyStopping(patience = 30, score_function = get_dice_score,
	                              trainer = trainer, min_delta = 0.01, cumulative_delta = True)
	evaluator.add_event_handler(event_name = Events.EPOCH_COMPLETED, handler = early_stopper)

	trainer.run()


def infer(data_folder = ".", model_folder = "runs", prediction_folder = "output"):
	"""
	run inference, the output folder will be "./output"
	"""
	print('start infer function')
	ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
	ckpt = ckpts[-1]
	for x in ckpts:
		logging.info(f"available model file: {x}.")
	logging.info("----")
	logging.info(f"using {ckpt}.")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = get_net().to(device)
	net.load_state_dict(torch.load(ckpt, map_location = device))
	net.eval()

	image_folder = os.path.abspath(data_folder)
	images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
	logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
	infer_files = [{"image": img} for img in images]

	keys = ("image",)
	infer_transforms = get_xforms("infer", keys)
	infer_ds = monai.data.Dataset(data = infer_files, transform = infer_transforms)
	infer_loader = monai.data.DataLoader(
		infer_ds,
		batch_size = 1,  # image-level batch to the sliding window method, not the window-level batch
		num_workers = 2,
		pin_memory = torch.cuda.is_available(),
	)

	inferer = get_inferer()
	saver = monai.data.NiftiSaver(output_dir = prediction_folder, mode = "nearest")
	with torch.no_grad():
		for infer_data in infer_loader:
			logging.info(
				f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
			preds = inferer(infer_data[keys[0]].to(device), net)
			n = 1.0
			for _ in range(4):
				_img = RandGaussianNoised(
					keys[0], prob = 1.0, std = 0.01)(infer_data)[keys[0]]
				pred = inferer(_img.to(device), net)
				preds = preds + pred
				n = n + 1.0
				for dims in [[2], [3]]:
					flip_pred = inferer(torch.flip(
						_img.to(device), dims = dims), net)
					pred = torch.flip(flip_pred, dims = dims)
					preds = preds + pred
					n = n + 1.0
			preds = preds / n
			preds = (preds.argmax(dim = 1, keepdims = True)).float()
			saver.save_batch(preds, infer_data["image_meta_dict"])

	# copy the saved segmentations into the required folder structure for submission)
	submission_dir = os.path.join(prediction_folder, "to_submit")
	if not os.path.exists(submission_dir):
		os.makedirs(submission_dir)
	files = glob.glob(os.path.join(prediction_folder, "*", "*.nii.gz"))
	for f in files:
		new_name = os.path.basename(f)
		to_name = os.path.join(submission_dir, new_name)
		shutil.copy(f, to_name)
	logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
	"""
	Usage:
		python DynUnet.py train --data_folder x --model_folder y --pred_folder z" # run the training pipeline
		python DynUnet.py infer --data_folder x --model_folder y --pred_folder z" # run the inference pipeline
	"""

	parser = argparse.ArgumentParser(description = "Run a basic UNet segmentation baseline.")
	parser.add_argument("mode", metavar = "mode", default = "train", choices = ("train", "infer"),
	                    type = str, help = "mode of workflow")
	parser.add_argument("--data_folder", default = "",
	                    type = str, help = "training data folder")
	parser.add_argument("--model_folder", default = "runs",
	                    type = str, help = "model folder")
	parser.add_argument("--pred_folder", default = "",
	                    type = str, help = "pred folder")
	args = parser.parse_args()

	monai.config.print_config()
	monai.utils.set_determinism(seed = 0)
	logging.basicConfig(stream = sys.stdout, level = logging.INFO)

	if args.mode == "train":
		data_folder = args.data_folder
		train(data_folder = data_folder, model_folder = args.model_folder)
	elif args.mode == "infer":
		data_folder = args.data_folder
		infer(data_folder = data_folder, model_folder = args.model_folder,
		      prediction_folder = args.pred_folder)
	else:
		raise ValueError("Unknown mode.")
