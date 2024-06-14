# Torch
from torch import nn

# Torchvision
from torchvision.models import resnet50, ResNet50_Weights

###############################################################################
class ResNet50Backbone(nn.Module):
	
	# -------------------------------------------------------------------------
	def __init__(self):
		super().__init__()
		
		self.resnet50_32s = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
		
		# Remove the fully connected layer
		self.resnet50_32s.fc = nn.Sequential()

		# Freeze batch norms
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.requires_grad = False
				m.bias.requires_grad = False
	
	# -------------------------------------------------------------------------
	def extract_features(self, x_input):
		self.resnet50_32s.eval()
		x = self.resnet50_32s.conv1(x_input)
		x = self.resnet50_32s.bn1(x)
		x = self.resnet50_32s.relu(x)
		x = self.resnet50_32s.maxpool(x)

		x = self.resnet50_32s.layer1(x)
		
		x_8s = self.resnet50_32s.layer2(x)
		x_16s = self.resnet50_32s.layer3(x_8s)
		x_32s = self.resnet50_32s.layer4(x_16s)

		return x_8s, x_16s, x_32s
