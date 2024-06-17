# Python
import numpy as np
import os

# TorchVision
from torchvision import transforms

# Pillow
from PIL import Image

# Pandas
import pandas as pd

# DeepFish
from .helpers import slice_df_reg

###############################################################################
class FishReg:
	
	# -------------------------------------------------------------------------
	def __init__(self, split, transform=None, datadir="", n_samples=None, habitat=None):

		self.split = split
		self.n_classes = 2
		self.datadir = datadir
		self.transform = transform

		self.img_names, self.labels, self.counts = self.__reg_data(datadir, split, habitat)

		if n_samples:
			self.img_names = self.img_names[:n_samples] 
			self.labels = self.labels[:n_samples] 
			self.counts = self.counts[:n_samples]

		self.path = self.datadir

	# -------------------------------------------------------------------------
	def __len__(self):
		return len(self.img_names)

	# -------------------------------------------------------------------------
	def __getitem__(self, index):
		name = self.img_names[index]
		image_pil = Image.open(self.path + "/images/"+ name + ".jpg")
	
		if self.transform:
			image = self.transform(image_pil)
		else:
			image = image_pil

		batch = {"images": image,
				 "labels": float(self.labels[index] > 0),
				 "image_original":transforms.ToTensor()(image_pil),
				 "counts": float(self.counts[index]), 
				 "meta": {"index": index,
						  "image_id": index,
						  "split": self.split}}

		return batch

	# -----------------------------------------------------------------------------
	def __reg_data(self, datadir, split, habitat=None):
		df = pd.read_csv(os.path.join(datadir,  '%s.csv' % split))
		df = slice_df_reg(df, habitat)

		img_names = np.array(df['ID'])
		counts = np.array(df['counts'])
		labels = np.array(df['labels'])

		return img_names, labels, counts
