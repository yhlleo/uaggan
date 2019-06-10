# uaggan
A Pytorch implementation of "Unsupervised Attention-Guided Image-to-Image Translation", NIPS 2018, [[Paper]](https://arxiv.org/pdf/1806.02311.pdf) | [[TF code]](https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation)

---------

### Installation

For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### Prepare Dataset:

 - Download a CycleGAN dataset (e.g. horse2zebra, apple2orange):

```
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```
 - Use your own dataset: put images from each domain at folder_a and folder_b respectively.

### Usage

 - Training

```
sh ./scripts/train_uaggan.sh <gpu_id>
```

or

```
nohup sh -u ./scripts/train_uaggan.sh <gpu_id> > uag.log &
```

- Test

```
sh ./scripts/test_uaggan.sh <gpu_id>
```

### Faster version

Combining the attention module and domain translation module into one generator.

//TODO

### Mapping results

 - horse2zebra (without early stopping, threshold=0.1)

|Real A|Attention Mask|Fake B|Real B|Attention Mask|Fake A|
|:----:|:----:|:----:|:----:|:----:|:----:|
|![](./figures/no-early-stopping/n02381460_1110_real_A.png)|![](./figures/no-early-stopping/n02381460_1110_att_A_viz.png)|![](./figures/no-early-stopping/n02381460_1110_masked_fake_B.png)|![](./figures/no-early-stopping/n02381460_1000_real_B.png)|![](./figures/no-early-stopping/n02381460_1000_att_B_viz.png)|![](./figures/no-early-stopping/n02381460_1000_masked_fake_A.png)|

[[More Results >>>]](./figures/Visualization-uag.md)

 - apple2orange


----

**Notes**: This project is based on the framework: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
