# python utils/make_dataset.py
python utils/make_config.py --model_name resnet50d resnet50d --batch_size 128 --out_features 6 --lr 0.0001 --img_size 384 384 --aug_ver 1 --loss_fn ib_focal --max_epoch 40 --devices 0 --mode fit_extract --fps_sampling 5 5

files="../result/model_name=resnet50d*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train.py --config "${filepath}/config.yaml" --use_wandb True 
    fi
done

python utils/make_config.py --model_name pgl_sum pgl_sum --loss_fn mae --lr 5e-5 --max_epoch 100 --monitor val_mae --ckpt_mode min --mode fit_test --feats_dir model_name=resnet50d-fps_sampling=5-img_size=384

files="../result/model_name=pgl_sum*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.ckpt"
        if [ -e $flag ] ; then
            continue
        fi

        python train_summarizer.py --config "${filepath}/config.yaml" --use_wandb True
    fi
done

python utils/make_config.py --model_name asformer asformer --lr 0.03 --max_epoch 10 --mode fit_test --module RIF RIF --channel_masking_rate 0.5 --out_features 2 --test_vid_idx 0 1 2 3 4 5 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 0 0 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=0-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 1 1 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=1-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 2 2 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=2-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 3 3 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=3-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 4 4 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=4-fps_sampling=5
python utils/make_config.py --model_name asformer asformer --lr 0.01  --max_epoch 20 --mode fit_test --module WR WR --channel_masking_rate 0.5 --out_features 8 8 --test_vid_idx 5 5 --feats_dir model_name=resnet50d-img_size=384-fps_sampling=5 --RIF_dir model_name=asformer-module=RIF-test_vid_idx=5-fps_sampling=5
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