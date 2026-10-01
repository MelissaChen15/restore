import torch

STEMS = ("vocals", "music", "broadband", "transient", "residual", "extension")


def load_model(checkpoint_path, device="cpu"):
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    klass = package["klass"]
    model = klass(*package["args"], **package["kwargs"])
    model.load_state_dict(package["state"])
    model.eval()
    return model.to(device)
