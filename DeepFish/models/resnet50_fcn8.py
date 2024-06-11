# Torch
import torch.nn as nn

# DeepFish
from . import resfcn

###############################################################################
class ResNet50FCN8(nn.Module):
	
	# -------------------------------------------------------------------------
	def __init__(self, n_classes=2):
		super().__init__()
		self.n_classes = n_classes

		self.backbone = resfcn.ResBackbone()

		resnet_block_expansion_rate = self.backbone.layer1[0].expansion


		self.score_32s = nn.Conv2d(512 * resnet_block_expansion_rate,
								   self.n_classes,
								   kernel_size=1)

		self.score_16s = nn.Conv2d(256 * resnet_block_expansion_rate,
								   self.n_classes,
								   kernel_size=1)

		self.score_8s = nn.Conv2d(128 * resnet_block_expansion_rate,
								  self.n_classes,
								  kernel_size=1)

	# -------------------------------------------------------------------------
	def forward(self, x):
		# 1. ResNet50 features extraction
		x_8s, x_16s, x_32s = self.backbone.extract_features(x)

		# 2. FCN8
		logits_8s = self.score_8s(x_8s)
		logits_16s = self.score_16s(x_16s)
		logits_32s = self.score_32s(x_32s)
		
		logits_16s_spatial_dim = logits_16s.size()[2:]
		logits_8s_spatial_dim = logits_8s.size()[2:]
		input_spatial_dim = x.size()[2:]

		logits_16s += nn.functional.interpolate(logits_32s,
												size=logits_16s_spatial_dim,
												mode="bilinear",
												align_corners=True)

		logits_8s += nn.functional.interpolate(logits_16s,
												size=logits_8s_spatial_dim,
												mode="bilinear",
												align_corners=True)

		logits_upsampled = nn.functional.interpolate(logits_8s,
													size=input_spatial_dim,
													mode="bilinear",
													align_corners=True)

		return logits_upsampled
