# MLCRNet

**MLCRNet: Multi-Level Context Refinement for Semantic
Segmentation in Aerial Images**


*Zhifeng Huang, Qian Zhang, Guixu Zhang*

[Remote Sensing 2022.3](https://www.mdpi.com/2072-4292/14/6/1498)

## Installation

*Note: We re-implemented MLCRNet based on MMSegmentation, we recommend using the **exact version** of the packages to avoid running issues.*

1. Install PyTorch 1.7.1 and torchvision 0.8.2 following the [official guide](https://pytorch.org/get-started/locally/).

2. This project depends on mmsegmentation 0.27.0 and mmcv 1.6.1, so you may follow its instructions to [setup environment](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md) and [prepare datasets](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).


## Evaluation

You may evaluate the model on single GPU by running:

```bash
python test.py \
	--config configs/mlcrnet/mlcrnet_r50-d16_512x512+80k_potsdam.py \
	--checkpoint /path/to/ckpt \
	--eval mIoU
```

To evaluate on multiple GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node 2 test.py \
	--launcher pytorch \
	--config configs/mlcrnet/mlcrnet_r50-d16_512x512+80k_potsdam.py \
	--checkpoint /path/to/ckpt 
	--eval mIoU
```

You may add `--aug-test` to enable multi-scale + flip evaluation. The `test.py` script is copy-pasted from mmsegmentation. Please refer to [this](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/inference.md) link for more usage (e.g., visualization).

## Training

We train all models on 2 2080ti GPUs. For example, to train MLCRNet_r50-d16, run:

```bash
python -m torch.distributed.launch --nproc_per_node 2 train.py 
	--launcher pytorch \
	--config configs/mlcrnet/mlcrnet_r50-d16_512x512+80k_potsdam.py \
	--work-dir /path/to/workdir
```

You may need to adjust `data.samples_per_gpu` if you plan to train on less GPUs. Please refer to [this](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/train.md) link for more training optioins.

## Citation

```
@article{huang2022mlcrnet,
  title={MLCRNet: Multi-Level Context Refinement for Semantic Segmentation in Aerial Images},
  author={Huang, Zhifeng and Zhang, Qian and Zhang, Guixu},
  journal={Remote Sensing},
  volume={14},
  number={6},
  pages={1498},
  year={2022},
  publisher={MDPI}
}
```

