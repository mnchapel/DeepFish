# Torch
import torch.nn as nn

# DeepFish
from .resnet50_backbone import ResNet50Backbone

###############################################################################
class ResNet50FCN8(nn.Module):
	
	# -------------------------------------------------------------------------
	def __init__(self, n_classes):
		super().__init__()

		# ResNet50 (backbone)
		self.backbone = ResNet50Backbone()

		# FCN8
		expansion_rate = self.backbone.layer1[0].expansion

		self.score_32s = nn.Conv2d(512 * expansion_rate, n_classes, kernel_size=1)
		self.score_16s = nn.Conv2d(256 * expansion_rate, n_classes, kernel_size=1)
		self.score_8s  = nn.Conv2d(128 * expansion_rate, n_classes, kernel_size=1)

	# -------------------------------------------------------------------------
	def forward(self, x):
		# 1. Extract ResNet50 features
		x_8s, x_16s, x_32s = self.backbone.extract_features(x)

		# 2. Get FCN8 output
		logits_32s = self.score_32s(x_32s)
		logits_16s = self.score_16s(x_16s)
		logits_8s = self.score_8s(x_8s)
		
		spatial_dim_16s = logits_16s.size()[2:]
		spatial_dim_8s = logits_8s.size()[2:]
		spatial_dim_x = x.size()[2:]

		logits_16s += self.upsampling(logits_32s, spatial_dim_16s)
		logits_8s += self.upsampling(logits_16s, spatial_dim_8s)
		logits_upsampled = self.upsampling(logits_8s, spatial_dim_x)

		return logits_upsampled

	# -------------------------------------------------------------------------
	def upsampling(self, logits, spatial_dim):
		return nn.functional.interpolate(logits, spatial_dim, mode="bilinear",	align_corners=True)
