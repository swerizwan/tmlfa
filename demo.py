import argparse

import cv2
import numpy as np
import torch

from model import build_model

@torch.no_grad()
def inference(cfg, weight, img):
    """
    Perform inference using the provided model configuration and weights.

    Args:
        cfg (SwinFaceCfg): Configuration object for the model.
        weight (str): Path to the model weights.
        img (str or None): Path to the input image or None for random input.
    """
    if img is None:
        # Generate a random image if no image path is provided
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        # Read and resize the image from the provided path
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    # Convert the image to RGB and transpose to match the model's input format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)  # Normalize the image

    # Build the model using the provided configuration
    model = build_model(cfg)
    dict_checkpoint = torch.load(weight)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()  # Set the model to evaluation mode
    result = model(img)  # Perform inference

    # Print the results
    for each in result.keys():
        print(each, "\t", result[each][0].numpy())


class SwinFaceCfg:
    """
    Configuration class for SwinFace model.

    This class contains the necessary parameters for building and configuring the SwinFace model.
    """
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for j in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512


if __name__ == "__main__":
    # Initialize the configuration
    cfg = SwinFaceCfg()

    # Define the argument parser
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='<your path>/checkpoint_step_79999_gpu_0.pt',
                        help='Path to the model weights')
    parser.add_argument('--img', type=str, default="<your path>/test.jpg",
                        help='Path to the input image')
    args = parser.parse_args()

    # Perform inference using the provided arguments
    inference(cfg, args.weight, args.img)