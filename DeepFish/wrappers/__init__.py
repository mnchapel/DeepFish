# DeepFish
from .clf_wrapper import ClfWrapper
from .reg_wrapper import RegWrapper
from .loc_wrapper import LocWrapper
from .seg_wrapper import SegWrapper

# -----------------------------------------------------------------------------
def get_wrapper(wrapper_name, model, opt=None):
	if wrapper_name == "clf_wrapper":
		return ClfWrapper(model, opt)

	if wrapper_name == "reg_wrapper":
		return RegWrapper(model, opt)

	if wrapper_name == "loc_wrapper":
		return LocWrapper(model, opt)

	if wrapper_name == "seg_wrapper":
		return SegWrapper(model, opt)
