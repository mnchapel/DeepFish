# Haven
from haven import haven_utils as hu

EXP_GROUPS = {
	"clf": {
		"dataset":["fish_clf"],
		"task":["clf"],
		"model":["resnet50_ffn"],
		"batch_size": [1],
		"transform":["resize_normalize"],
		"max_epoch": [1000],
		"wrapper":["clf_wrapper"]
	},
	"reg": {
		"dataset":["fish_reg"],
		"task":["reg"],
		"model":["resnet50_ffn"],
		"batch_size": [1],
		"transform":["resize_normalize"],
		"max_epoch": [1000],
		"wrapper":["reg_wrapper"]
	},
	"loc": {
		"dataset":["fish_loc"],
		"task": ["loc"],
		"model":["resnet50_fcn8"],
		"batch_size": [1],
		"transform":["rgb_normalize"],
		"max_epoch": [1000],
		"wrapper":["loc_wrapper"]
	},
	"seg": {
		"dataset":["fish_seg"],
		"task": ["seg"],
		"model":["resnet50_fcn8"],
		"batch_size": [1],
		"transform":["rgb_normalize"],
		"max_epoch": [1000],
		"wrapper":["seg_wrapper"]
	},
}

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
