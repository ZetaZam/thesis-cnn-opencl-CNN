import torch
import torchvision.models as models
import numpy as np

def prepare_weights_pretrained_npz(output_path="alexnet_weights_pretrained.npz"):
    model = models.alexnet(pretrained=True)
    weights_dict = {}

    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()

    np.savez(output_path, **weights_dict)
    print(f"Pretrained weights saved to {output_path}")

if __name__ == "__main__":
    prepare_weights_pretrained_npz()
