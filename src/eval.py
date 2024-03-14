import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import random
from tqdm import tqdm
import shutil
import ffmpeg

import datetime
from logging import INFO, basicConfig, getLogger
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import random

logger = getLogger(__name__)

def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for /// with /// Dataset.
        """
    )

    parser.add_argument(
        "--n_sec",
        required=False,
        type=int,
        default=5,
        help="random seed",
    )
    return parser.parse_args()
    
def segment_bars_with_confidence(save_path, confidence, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('gist_rainbow')
 
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    interval = 1 / (num_pics+1)
    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
        ax1.imshow([label], **barprops)
 
    ax4 = fig.add_axes([0, interval, 1, interval])
    ax4.set_xlim(0, len(confidence))
    ax4.set_ylim(0, 1)
    ax4.plot(range(len(confidence)), confidence)
    ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def segment_bars(color, save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap(color) #https://beiznotes.org/matplot-cmap-list/
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)
 
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()

def get_gts(x):
    if x == 0:
        return 0
    elif x == 0.5:
        return 10
    else:
        return 20
    
def main():
    args = get_arguments()
    
    n_sec = args.n_sec
    random_p = []
    random_n = []
    uni_p = []
    uni_n = []
    std_p = []
    std_n = []
    std_ideal_p = []
    std_ideal_n = []
    phase_accs = []
    phase_f1s = []
    di_metrics_uniform = []
    di_metrics_std = []
    di_metrics_std_ideal = []
    di_metrics_random = []
    
    os.makedirs(f'../outputs/n_sec={n_sec}', exist_ok=True)
    logname = os.path.join(f"../outputs/n_sec={n_sec}", f"{datetime.datetime.now():%Y-%m-%d}_eval.log")
    basicConfig(
        level=INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    for video_idx in range(6):    
        result_path = f"../result/model_name=pgl_sum-lr=5e-06-test_vid_idx={video_idx}"
        save_path = f"../outputs/n_sec={n_sec}/video_idx={video_idx}"
        os.makedirs(save_path, exist_ok=True)
        gts = np.load(os.path.join(result_path, 'gts.npy'))
        out = np.load(os.path.join(result_path, 'outputs.npy'))
        
        df = pd.DataFrame({'gts':gts[0],'out':out[0]})
        df['id'] = df.index // n_sec
        max_id = df.id.max()
        df = df.merge(df.groupby('id').aggregate('mean').rename(columns={'out':'out_mean'})['out_mean'], on='id', how='left')
        
        logger.info(f'video{video_idx}')
        logger.info(f'  mae: {mae(df.gts, df.out)}')
        logger.info(f'  mse: {mse(df.gts, df.out)}')
        
        th = df.sort_values('out_mean',ascending=False).iloc[len(df)//10].out_mean
        df['summary'] = df.out_mean.map(lambda x: 1 if x >= th else 0)
        phase_df = pd.read_csv('../result/model_name=resnet50d-img_size=384/processed_df.csv')
        phase_df = phase_df[phase_df.video_idx==video_idx].reset_index()
        phase_df['p_phase'] = np.load(f'../result/model_name=asformer_2-out_features=8-test_vid_idx={video_idx}/preds.npy')
        tmp = pd.concat([df, phase_df[['phase','p_phase','file_name']]], axis=1)    
        
        dic = {
            'design': 0,
            'anesthesia': 1,
            'incision': 2,
            'hemostasis': 3,
            'dissection': 4,
            'closure': 5,
            'others': 6,
            'irrelevant': 7
        }
        
        tmp['phase'] = tmp.phase.map(lambda x: dic[x]) 
        logger.info(f'acc 8classes: {accuracy_score(tmp.phase, tmp.p_phase)}')
        logger.info(f'f1 8classes: {f1_score(tmp.phase, tmp.p_phase, average="macro")}')
        tmp['phase_7'] = tmp.phase.replace(7, np.nan).ffill().bfill()
        tmp['p_phase_7'] = tmp.p_phase.replace(7, np.nan).ffill().bfill()
        logger.info(f'acc 7classes: {accuracy_score(tmp.phase, tmp.p_phase)}')
        phase_accs.append(accuracy_score(tmp.phase, tmp.p_phase))
        logger.info(f'f1 7classes: {f1_score(tmp.phase, tmp.p_phase, average="macro")}')
        phase_f1s.append(f1_score(tmp.phase, tmp.p_phase, average="macro"))
                
        li = []
        cnt = 0
        p = tmp.p_phase.values[0]
        for i in tqdm(range(len(tmp))):
            if p == tmp.p_phase.values[i]:
                li.append(cnt)
            else:
                cnt += 1
                p = tmp.p_phase.values[i]
                li.append(cnt)
                
        tmp['phase_idx'] = li
        
        li = []
        cnt = 0
        p = tmp.phase.values[0]
        for i in tqdm(range(len(tmp))):
            if p == tmp.phase.values[i]:
                li.append(cnt)
            else:
                cnt += 1
                p = tmp.phase.values[i]
                li.append(cnt)
                
        tmp['phase_ideal_idx'] = li        

        df_std = tmp[['out_mean','phase_idx']].groupby('phase_idx').aggregate(['min','max','std','mean'])['out_mean'].reset_index(drop=True)
        tmp = tmp.merge(df_std, left_on='phase_idx', right_index=True, how='left')
        tmp['std'] = tmp['std'].fillna(1)  
        tmp['out_norm_minmax'] = (tmp['out_mean'] - tmp['min']) / (tmp['max'] - tmp['min'])
        tmp['out_norm_std'] = (tmp['out_mean'] - tmp['mean']) / tmp['std']
        tmp.loc[tmp.p_phase==7, ['out_norm_minmax']] = -100
        tmp.loc[tmp.p_phase==7, ['out_norm_std']] = -100
        th = tmp.sort_values('out_norm_minmax',ascending=False).iloc[len(tmp)//10].out_norm_minmax
        tmp['summary_norm_minmax'] = tmp.out_norm_minmax.map(lambda x: 1 if x >= th else 0)
        th = tmp.sort_values('out_norm_std',ascending=False).iloc[len(tmp)//10].out_norm_std
        tmp['summary_norm_std'] = tmp.out_norm_std.map(lambda x: 1 if x >= th else 0)
        tmp.drop(['min', 'max', 'mean', 'std'], axis=1, inplace=True)

        #ideal
        df_std = tmp[['out_mean','phase_ideal_idx']].groupby('phase_ideal_idx').aggregate(['min','max','std','mean'])['out_mean'].reset_index(drop=True)
        tmp = tmp.merge(df_std, left_on='phase_ideal_idx', right_index=True, how='left')
        tmp['std'] = tmp['std'].fillna(1)
        tmp['out_norm_minmax_ideal'] = (tmp['out_mean'] - tmp['min']) / (tmp['max'] - tmp['min'])
        tmp['out_norm_std_ideal'] = (tmp['out_mean'] - tmp['mean']) / tmp['std']
        tmp.loc[tmp.phase==7, ['out_norm_minmax_ideal']] = -100
        tmp.loc[tmp.phase==7, ['out_norm_std_ideal']] = -100
        th = tmp.sort_values('out_norm_minmax_ideal',ascending=False).iloc[len(tmp)//10].out_norm_minmax_ideal
        tmp['summary_norm_minmax_ideal'] = tmp.out_norm_minmax_ideal.map(lambda x: 1 if x >= th else 0)
        th = tmp.sort_values('out_norm_std_ideal',ascending=False).iloc[len(tmp)//10].out_norm_std_ideal
        tmp['summary_norm_std_ideal'] = tmp.out_norm_std_ideal.map(lambda x: 1 if x >= th else 0)
        

        segment_bars('bwr', os.path.join(save_path,'gts.png'), tmp.gts.map(lambda x: get_gts(x)).values)
        print('uniform')
        segment_bars('binary', os.path.join(save_path,'uniform.png'), 256 - tmp.summary.values*256)
        print('standardization')
        segment_bars('binary', os.path.join(save_path,'std.png'), 256 - tmp.summary_norm_std.values*256)
        segment_bars('gist_rainbow', os.path.join(save_path,'phase_gts.png'), tmp.phase.values*2.5)
        segment_bars('gist_rainbow', os.path.join(save_path,'phase_preds.png'), tmp.p_phase.values*2.5)
        
        logger.info(f"  Uniform {tmp[tmp.summary==1].phase.value_counts()}")
        segment_bars('gist_rainbow', os.path.join(save_path,'output_phase_std.png'), tmp[tmp.summary==1].phase.values*2.5)
        logger.info(f"  Std {tmp[tmp.summary_norm_std==1].phase.value_counts()}")
        output = tmp[tmp.summary==1]
        segment_bars('gist_rainbow', os.path.join(save_path,'output_phase_uniform.png'), tmp[tmp.summary_norm_std==1].phase.values*2.5)
        
        
        #random
        
        random_p_tmp = []
        random_n_tmp = []
        di_metrics = []
        di_metrics_tmp = []
        for i in range(10):
            random_list = [_ for _ in range(tmp['id'].max())]
            random_list = random.sample(random_list, len(random_list)//10)
            # random_list = [random.randint(0,int(tmp['id'].max())) for _ in range(int(tmp['id'].max()) // 10)]
            tmp['random_summary'] = tmp['id'].map(lambda x: 1 if x in random_list else 0)
            output = tmp[tmp.random_summary == 1]
            random_p_tmp.append(len(output[output.gts==1]) / len(output))
            random_n_tmp.append(len(output[output.gts==0]) / len(output))
            for j in range(8):
                if j != 7:
                    gts_phase_rate = len(tmp[tmp.phase==j]) / len(tmp[tmp.phase!=7])
                    pred_phase_rate = len(output[output.phase==j]) / len(output)
                    di_metrics_tmp.append(np.abs(gts_phase_rate - pred_phase_rate))
                else:
                    gts_phase_rate = 0
                    pred_phase_rate = len(output[output.phase==j]) / len(output)
                    di_metrics_tmp.append(np.abs(gts_phase_rate - pred_phase_rate))
            di_metrics.append(np.mean(di_metrics_tmp))
        logger.info(f'di_metrics random: {np.mean(di_metrics)}')
        di_metrics_random.append(np.mean(di_metrics))
        random_p.append(np.mean(random_p_tmp))
        random_n.append(np.mean(random_n_tmp))
        logger.info(f'  random p_rate: {np.mean(random_p_tmp)}, n_rate: {np.mean(random_n_tmp)}')
        
        
        
        # Uniform
        p_rate = len(output[output.gts==1]) / len(output)
        uni_p.append(p_rate)
        n_rate = len(output[output.gts==0]) / len(output)
        uni_n.append(n_rate)
        logger.info(f'  Uniform p_rate: {p_rate}, n_rate: {n_rate}')
        
        
        di_metrics = []
        for i in range(8):
            if i != 7:
                gts_phase_rate = len(tmp[tmp.phase==i]) / len(tmp[tmp.phase!=7])
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
            else:
                gts_phase_rate = 0
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
        logger.info(f'di_metrics uniform: {np.mean(di_metrics)}')
        di_metrics_uniform.append(np.mean(di_metrics))
        
        video_path = tmp[tmp.summary==1].file_name.values
        os.makedirs('../outputs/images', exist_ok=True)
        for i in range(len(video_path)):
            s = str(i).zfill(6)
            copypath = f"../SummarizationDataset/video_split/video0{video_idx}/"+video_path[i]
            savepath = (
                '../outputs/images/'
                + s
                + ".png"
            )
            shutil.copyfile(copypath, savepath)

        stream = ffmpeg.input(
            "../outputs/images/%6d.png",r=1
        )
        stream = ffmpeg.output(
            stream,
            f"../outputs/n_sec={n_sec}/video_idx={video_idx}/video0{video_idx}_sum_uniform.mp4",
            r=1
        )
        ffmpeg.run(stream)
        shutil.rmtree('../outputs/images')
        
        
        # STD
        output = tmp[tmp.summary_norm_minmax==1]
        p_rate = len(output[output.gts==1]) / len(output)
        std_p.append(p_rate)
        n_rate = len(output[output.gts==0]) / len(output)
        std_n.append(n_rate)
        logger.info(f'  std p_rate: {p_rate}, n_rate: {n_rate}')
        
        di_metrics = []
        for i in range(8):
            if i != 7:
                gts_phase_rate = len(tmp[tmp.phase==i]) / len(tmp[tmp.phase!=7])
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
            else:
                gts_phase_rate = 0
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
        logger.info(f'di_metrics std: {np.mean(di_metrics)}')
        di_metrics_std.append(np.mean(di_metrics))
        
        video_path = tmp[tmp.summary_norm_std==1].file_name.values
        os.makedirs('../outputs/images', exist_ok=True)
        for i in range(len(video_path)):
            s = str(i).zfill(6)
            copypath = f"../SummarizationDataset/video_split/video0{video_idx}/"+video_path[i]
            savepath = (
                '../outputs/images/'
                + s
                + ".png"
            )
            shutil.copyfile(copypath, savepath)

        stream = ffmpeg.input(
            "../outputs/images/%6d.png",r=1
        )
        stream = ffmpeg.output(
            stream,
            f"../outputs/n_sec={n_sec}/video_idx={video_idx}/video0{video_idx}_sum_std.mp4",
            r=1
        )
        ffmpeg.run(stream)
        shutil.rmtree('../outputs/images')

        # STD Ideal
        output = tmp[tmp.summary_norm_minmax_ideal==1]
        p_rate = len(output[output.gts==1]) / len(output)
        std_ideal_p.append(p_rate)
        n_rate = len(output[output.gts==0]) / len(output)
        std_ideal_n.append(n_rate)
        logger.info(f'  std ideal p_rate: {p_rate}, n_rate: {n_rate}')
        
        di_metrics = []
        for i in range(8):
            if i != 7:
                gts_phase_rate = len(tmp[tmp.phase==i]) / len(tmp[tmp.phase!=7])
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
            else:
                gts_phase_rate = 0
                pred_phase_rate = len(output[output.phase==i]) / len(output)
                di_metrics.append(np.abs(gts_phase_rate - pred_phase_rate))
        logger.info(f'di_metrics std ideal: {np.mean(di_metrics)}')
        di_metrics_std_ideal.append(np.mean(di_metrics))
        
        video_path = tmp[tmp.summary_norm_std_ideal==1].file_name.values
        os.makedirs('../outputs/images', exist_ok=True)
        for i in range(len(video_path)):
            s = str(i).zfill(6)
            copypath = f"../SummarizationDataset/video_split/video0{video_idx}/"+video_path[i]
            savepath = (
                '../outputs/images/'
                + s
                + ".png"
            )
            shutil.copyfile(copypath, savepath)

        stream = ffmpeg.input(
            "../outputs/images/%6d.png",r=1
        )
        stream = ffmpeg.output(
            stream,
            f"../outputs/n_sec={n_sec}/video_idx={video_idx}/video0{video_idx}_sum_std_ideal.mp4",
            r=1
        )
        ffmpeg.run(stream)
        shutil.rmtree('../outputs/images')
        tmp.to_csv(f"../outputs/n_sec={n_sec}/out_df_{video_idx}.csv")


    logger.info(f'random positive mean {np.mean(random_p)}')
    logger.info(f'random negative mean {np.mean(random_n)}')
    logger.info(f'uniform positive mean {np.mean(uni_p)}')
    logger.info(f'uniform negative mean {np.mean(uni_n)}')
    logger.info(f'std positive mean {np.mean(std_p)}')
    logger.info(f'std negative mean {np.mean(std_n)}')
    logger.info(f'std_ideal positive mean {np.mean(std_ideal_p)}')
    logger.info(f'std_ideal negative mean {np.mean(std_ideal_n)}')    
    logger.info(f'phase acc mean {np.mean(phase_accs)}')
    logger.info(f'phase f1 mean {np.mean(phase_f1s)}')
    logger.info(f'di metrics random {np.mean(di_metrics_random)}')
    logger.info(f'di metrics uniform {np.mean(di_metrics_uniform)}')
    logger.info(f'di metrics std {np.mean(di_metrics_std)}')
    logger.info(f'di metrics std_ideal {np.mean(di_metrics_std_ideal)}')
    
    hmean_random = stats.hmean([np.mean(random_p) - np.mean(random_n), np.mean(di_metrics_random)])
    hmean_uniform = stats.hmean([np.mean(uni_p) - np.mean(uni_n), np.mean(di_metrics_uniform)])
    hmean_std = stats.hmean([np.mean(std_p) - np.mean(std_n), np.mean(di_metrics_std)])
    hmean_std_ideal = stats.hmean([np.mean(std_ideal_p) - np.mean(std_ideal_n), np.mean(di_metrics_std_ideal)])
    logger.info(f'hmean random {hmean_random}')
    logger.info(f'hmean uniform {hmean_uniform}')
    logger.info(f'hmean std {hmean_std}')
    logger.info(f'hmean std_ideal {hmean_std_ideal}')
    
        
if __name__ == '__main__':
    main()

