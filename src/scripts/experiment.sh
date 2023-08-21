# python utils/make_dataset.py
python utils/make_config.py --batch_size 128 128 --out_features 6 6 --lr 0.0001 0.0001 --fps_sampling 1 1 --img_size 224 224 --aug_ver 1 1 --loss_fn ib_focal ib_focal --max_epoch 20 20 --devices 0 --mode extract

files="../result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train.py --config "${filepath}/config.yaml" --use_wandb True 
    fi
done