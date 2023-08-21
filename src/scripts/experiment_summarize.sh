# python utils/make_dataset.py
python utils/make_config.py --model_name pgl_sum pgl_sum --loss_fn mse --lr 5e-5 5e-5 --monitor val_mse --ckpt_mode min

files="../result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train_summarizer.py --config "${filepath}/config.yaml" --use_wandb True
    fi
done