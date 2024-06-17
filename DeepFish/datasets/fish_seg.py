# Python
import numpy as np
import os

# Torch
import torch
from torchvision.io import read_image, ImageReadMode

# Pillow
from PIL import Image

# Pandas
import pandas as pd

# DeepFish
from .helpers import slice_df_reg

###############################################################################
class FishSeg:
	
	# -------------------------------------------------------------------------
	def __init__(self, split, transform=None, datadir="", n_samples=None, habitat=None):
		self.split = split
		self.n_classes = 2
		self.datadir = datadir
		self.transform = transform

		self.img_names, self.labels, self.mask_names = self.__seg_data(datadir, split, habitat)

		if n_samples:
			self.img_names = self.img_names[:n_samples] 
			self.mask_names = self.mask_names[:n_samples] 
			self.labels = self.labels[:n_samples]

		self.path = self.datadir

	# -------------------------------------------------------------------------
	def __len__(self):
		return len(self.img_names)

	# -------------------------------------------------------------------------
	def __getitem__(self, index):
		name = self.img_names[index]
		image = Image.open(self.path + "/images/"+ name + ".jpg")
	
		if self.transform:
			image = self.transform(image)

		mask_classes = Image.open(self.path + "/masks/"+ self.mask_names[index] + ".png").convert('L')

		mask_classes = torch.from_numpy(np.array(mask_classes)).float() / 255.
		batch = {"images": image,
				 "labels": self.labels[index],
				 "mask_classes": mask_classes,
				 "meta": {"index": index,
						  "image_id": index,
						  "split": self.split
						 }
				}

		return batch

	# -----------------------------------------------------------------------------
	def __seg_data(self, datadir, split,  habitat=None):
		df = pd.read_csv(os.path.join(datadir,  '%s.csv' % split))
		df = slice_df_reg(df, habitat)

		img_names = np.array(df['ID'])
		mask_names = np.array(df['ID'])
		labels = np.array(df['labels'])
		
		return img_names, labels, mask_names
