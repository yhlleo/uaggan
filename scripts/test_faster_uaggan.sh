GPU_ID=$1

python3.5 test.py \
  --gpu_ids ${GPU_ID} \
  --dataroot ./datasets/horse2zebra \
  --name faster_uaggan_horse2zebra \
  --model faster_uag_gan \
  --phase test \
  --num_test 500 \
  --thresh 0.1
