GPU_ID=0

python test.py \
  --dataroot ./datasets/horse2zebra \
  --name uaggan_horse2zebra \
  --model uag_gan \
  --phase test \
  --no_dropout
