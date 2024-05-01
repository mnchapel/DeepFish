# Torch
import torch
from torch import nn

# DeepFish
from . import resfcn

###############################################################################
class ResNet(torch.nn.Module):

	# -------------------------------------------------------------------------
	def __init__(self, n_classes=1):
		super().__init__()
		  
		# Features 
		self.n_outputs = n_outputs = 1
		
		# Backbone
		self.backbone = resfcn.ResBackbone()
		layers = list(map(int, str("100-100").split("-")))
		layers = [401408] + layers
		n_hidden = len(layers) - 1

		layerList = []      
		for i in range(n_hidden): 
			layerList += [nn.Linear(layers[i], layers[i+1]), nn.ReLU()]
		
		layerList += [nn.Linear(layers[i+1], n_outputs)]
		self.mlp = nn.Sequential(*layerList)

		# Freeze batch norms
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.requires_grad = False
				m.bias.requires_grad = False

	# -------------------------------------------------------------------------
	def forward(self, x):
		n = x.shape[0]
		logits_32s, logits_16s, logits_8s = self.backbone.extract_features(x)

		# 1. Extract ResNet features
		x = logits_32s.view(n, -1)
	   
		# 2. Get MLP output
		x = self.mlp(x)
		return x 
