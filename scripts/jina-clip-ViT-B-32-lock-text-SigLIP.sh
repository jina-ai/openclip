export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 -m training.main \
    --train-data="/home/jinaai/datasets/laion-400m/train-part-1/{00002..08100}.tar::/home/jinaai/datasets/laion-400m/train-part-2/{00000..02100}.tar" \
    --train-num-samples 24576000 \
    --val-data="/home/jinaai/datasets/laion-400m/train-part-1/{00000..00001}.tar" \
    --val-num-samples 15000 \
    --dataset-type webdataset \
    --batch-size 1024 \
    --warmup 10000 \
    --epochs 10 \
    --lr 5e-4 \
    --precision amp \
    --workers 2 \
    --model "jina-clip-ViT-B-32" \
    --force-custom-text \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --siglip \
    --log-every-n-steps 20 \
    --report-to "wandb" \
    --name "jina-clip-ViT-B-32-lock-text-SigLIP" \
    --wandb-project-name "jina-clip-laion400m-full" \
    --clip-benchmark-frequency 1 \
    --mteb-frequency 1 \
    --evaluate-on-start