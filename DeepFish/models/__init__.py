# DeepFish
from . import resnet, resnet50_fcn8

# -----------------------------------------------------------------------------
def get_model(model_name, exp_dict):
    if exp_dict['dataset'] == 'fish_loc':
            n_classes = 1
    else:
        n_classes = 2

    if model_name == "resnet50_fcn8":
        model = resnet50_fcn8.ResNet50FCN8(n_classes)

    if model_name == "resnet":
        model = resnet.ResNet(n_classes=1)

    return model
