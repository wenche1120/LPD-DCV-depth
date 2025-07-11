# Self-supervised Monocular Depth Estimation based on Laplacian Pyramid Decoder and Distance Cost Volume



## Training

To train on KITTI, run:

```bash
python train.py ./args_files/resnet_liteQTR_640.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

## Evaluation

To evaluate a model on KITTI, run:

```bash
python evaluate_lite_depth_config.py ./args_files/resnet_liteQTR_640.txt
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

## Inference with your own iamges

```bash
python test_simple.py --image_path --model_name
```
