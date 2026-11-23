from .iresnet import iresnet18  # , iresnet34, iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    else:
        raise ValueError()
