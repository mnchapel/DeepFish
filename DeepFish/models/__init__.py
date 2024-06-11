# DeepFish
from . import resnet, fcn8

# -----------------------------------------------------------------------------
def get_model(model_name, exp_dict):
    if exp_dict['dataset'] == 'fish_loc':
            n_classes = 1
    else:
        n_classes = 2

    if model_name == "fcn8":
        model = fcn8.FCN8(n_classes)

    if model_name == "resnet":
        model = resnet.ResNet(n_classes=1)

    return model
