# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


# --image_path
# --model_name

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    return parser.parse_args()


def test_simple(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    model_path = os.path.join("/home/model", args.model_name)

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    print("Loading pretrained encoder")
    encoder = networks.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)

    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=16, dim_out=64, embedding_dim=32,
                                                        query_nums=64, num_heads=4, min_val=0.001, max_val=80.0)
    # depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=32, dim_out=128, embedding_dim=32,
    #                                                     query_nums=128, num_heads=4, min_val=0.001, max_val=80.0)

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        base_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        # output_directory = args.image_path
        base_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    save_directory = os.path.join(base_directory, args.model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            input_image = input_image.to(device)
            print(input_image.shape, " shape")
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            output_name = os.path.splitext(os.path.basename(image_path))[0]

            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(save_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(save_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            vmin = np.percentile(disp_resized_np, 0)
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # 映射为彩色
            orig_cmap = cm.get_cmap('magma_r')
            new_cmap = cm.colors.LinearSegmentedColormap.from_list(
                'custom_magma',
                orig_cmap(np.linspace(0.1, 0.95, 256))
            )

            mapper = cm.ScalarMappable(norm=normalizer, cmap=new_cmap)
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(save_directory, "{}_disp.png".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)