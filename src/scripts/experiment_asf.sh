# python utils/make_dataset.py
python utils/make_config.py --model_name asformer asformer --batch_size 1 --lr 0.001 --fps_sampling 1 --loss_fn asf_loss --max_epoch 50 50 --mode fit_test

files="../result/model_name=asformer*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train_asf.py --config "${filepath}/config.yaml" --use_wandb True 
    fi
done