# uaggan
A Pytorch implementation of "Unsupervised Attention-Guided Image-to-Image Translation", [[Paper]](https://arxiv.org/pdf/1806.02311.pdf) | [[TF code]](https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation)

Based on the framework: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---------

### Installation

For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### Prepare Dataset:

 - Download a CycleGAN dataset (e.g. horse2zebra, apple2orange):

```
bash ./download_cyclegan_dataset.sh horse2zebra
```
 - Use your own dataset: put images from each domain at folder_a and folder_b respectively.

### Training

```
sh ./scripts/train_uaggan.sh
```

or

```
nohup sh -u ./scripts/train_uaggan.sh > uag.log &
```

