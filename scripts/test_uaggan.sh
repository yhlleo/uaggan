GPU_ID=$1

python3.5 test.py \
  --dataroot ./datasets/horse2zebra \
  --name uaggan_horse2zebra \
  --model uag_gan \
  --phase test \
  --num_test 10000
