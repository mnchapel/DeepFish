# Python
from enum import Enum

# DeepFish
from .fish_clf import FishClf
from .fish_reg import FishReg
from .fish_loc import FishLoc
from .fish_seg import FishSeg
from .helpers import get_transformer

###############################################################################
class DatasetDir(Enum):
	CLASSIFICATION = "/Classification/"
	LOCALIZATION = "/Localization/"
	SEGMENTATION = "/Segmentation/"

# -----------------------------------------------------------------------------
def get_dataset(split, exp_dict, datadir=None):
	dataset_name = exp_dict["dataset"]
	transform = get_transformer(exp_dict["transform"])
	n_samples = None
	habitat = None

	if "n_samples" in exp_dict:
		n_samples = exp_dict["n_samples"]
	if "habitat" in exp_dict:
		habitat = exp_dict["habitat"]

	match dataset_name:
		case "fish_clf":
			datadir = datadir + DatasetDir.CLASSIFICATION.value
			fish_dataset = FishClf
		case "fish_reg":
			datadir = datadir + DatasetDir.LOCALIZATION.value
			fish_dataset = FishReg
		case "fish_loc":
			datadir = datadir + DatasetDir.LOCALIZATION.value
			fish_dataset = FishLoc
		case "fish_seg":
			datadir = datadir + DatasetDir.SEGMENTATION.value
			fish_dataset = FishSeg
		case _:
			raise Exception(f"Exception in get_dataset(...) function: the dataset name {dataset_name} is not recognized. Please select fish_clf, fish_reg, fish_loc or fish_seg.")

	return fish_dataset(split, transform, datadir, n_samples, habitat)
