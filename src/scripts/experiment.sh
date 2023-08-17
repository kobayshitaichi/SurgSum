# python utils/make_dataset.py
python utils/make_config.py --batch_size 16 16

files="../result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train.py --config "${filepath}/config.yaml"
    fi
done