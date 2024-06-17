# DeepFish
from . import resnet50_fcn8, resnet50_ffn

# -----------------------------------------------------------------------------
def get_model(exp_dict):
	model_name = exp_dict["model"]
	dataset = exp_dict["dataset"]

	match model_name:
		case "resnet50_ffn":
			model = resnet50_ffn.ResNet50FFN()
		case "resnet50_fcn8":
			n_classes = 2

			if dataset == "fish_loc":
				n_classes = 1

			model = resnet50_fcn8.ResNet50FCN8(n_classes)

	return model
