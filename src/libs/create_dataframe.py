import pandas as pd
from pathlib import Path
import os
from logging import getLogger

logger = getLogger(__name__)

def get_dataframe(config):
    csv_path = Path(config.dataset_dir) / "csv"
    dirs = sorted(list(csv_path.glob('*csv')))
    
    all_df = pd.DataFrame()
    for i, path in enumerate(dirs):
        file_name = path.stem
        tmp = pd.read_csv(path)

        tmp['video_idx'] = int(file_name[-2:])  
        img_path = os.path.join(config.dataset_dir, 'video_split', file_name)
        tmp = tmp.iloc[:min(len(os.listdir(img_path)), len(tmp))]
        tmp['file_name'] = sorted(os.listdir(img_path)[:len(tmp)])
        
        if int(file_name[-2:]) in config.val_vid_idx:
            tmp['stage'] = 'val'
            factor = int(30 / config.fps_sampling_test)
        else:
            tmp['stage'] = 'train'
            factor = int(30 / config.fps_sampling)
        tmp = tmp.iloc[::factor]
        logger.info(f'{file_name} after subsampling: {len(tmp)}')
        all_df = pd.concat([all_df, tmp], axis=0)
        
    return all_df

