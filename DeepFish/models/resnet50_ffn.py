# Torch
import torch
from torch import nn

# DeepFish
from .resnet50_backbone import ResNet50Backbone

###############################################################################
class ResNet50FFN(torch.nn.Module):

	# -------------------------------------------------------------------------
	def __init__(self):
		super().__init__()
		
		# Features
		n_classes = 1
		
		# ResNet50 (backbone)
		self.backbone = ResNet50Backbone()

		## FFN
		layers = list(map(int, str("100-100").split("-")))
		layers = [100352] + layers # 2048*7*7 = 100,352
		n_hidden = len(layers) - 1

		layerList = []
		for i in range(n_hidden):
			layerList += [nn.Linear(layers[i], layers[i+1]), nn.ReLU()]
		
		layerList += [nn.Linear(layers[i+1], n_classes)]
		self.ffn = nn.Sequential(*layerList)

	# -------------------------------------------------------------------------
	def forward(self, x):
		# 1. Extract ResNet50 features
		_, _, x_32s= self.backbone.extract_features(x)
		n = x.shape[0]
		x = x_32s.view(n, -1)
	   
		# 2. Get FFN output
		x = self.ffn(x)

		return x 
