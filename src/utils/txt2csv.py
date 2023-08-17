import pandas as pd

import argparse
from tqdm import tqdm
import os
from pathlib import Path
import datetime

def main():
    args = get_arguments()
    ds_dir = Path(args.ds_dir)
    save_dir = Path(args.save_dir)
    fps = args.fps
    
    input_path = ds_dir  / "annotation"
    output_path = save_dir  / "csv"
    txt2csv(input_path, output_path)
    
def txt2csv(input_path, output_path):
    dirs = list(input_path.glob('*txt'))

    for num, txt_path in enumerate(tqdm(dirs)):
        file_name = txt_path.stem
        out_folder = output_path / file_name
        if os.path.exists(out_folder):
            continue
        print(f'{file_name} start')

        df = pd.read_table(dirs[num],comment='#')
        df.rename(columns={'開始時間 - 秒.ミリ秒':'start','終了時間 - 秒.ミリ秒':'end'},inplace=True)

        num_flames = int(df['end'].max() * 30 // 1)
        dic = {'Frame':[i for i in range(num_flames)], 'time':[0]*num_flames, 'field':[0]*num_flames,
        'phase':['others']*num_flames,'summary':[0]*num_flames}
        
        df_sf = df[df['Surgical Field'].notnull()]
        df_phase = df[df['Phase'].notnull()]
        df_sum = df[df['Summary'].notnull()]
        
        for i in range(num_flames):
            dic['time'][i] = str(datetime.timedelta(seconds=i/30))[:10]
        for i in range(len(df_sf)):
            start_frame = int(df_sf['start'].iloc[i] * 30)
            end_frame = int(df_sf['end'].iloc[i] * 30)

            for j in range(start_frame,end_frame):
                dic['field'][j] = df_sf['Surgical Field'].iloc[i]
                
        for i in range(len(df_phase)):
            start_frame = int(df_phase['start'].iloc[i] * 30)
            end_frame = int(df_phase['end'].iloc[i] * 30)

            for j in range(start_frame,end_frame):
                dic['phase'][j] = df_phase['Phase'].iloc[i]
                
        for i in range(len(df_sum)):
            start_frame = int(df_sum['start'].iloc[i] * 30)
            end_frame = int(df_sum['end'].iloc[i] * 30)

            for j in range(start_frame,end_frame):
                dic['summary'][j] = df_sum['Summary'].iloc[i]
                
        data = pd.DataFrame(dic)
        data['time'] = data['time'].map(lambda x: x if len(x) == 10 else x + '.00')
        data.loc[data[data.field==False].index,['phase']] = 'irrelevant'
        data.to_csv(output_path/Path(file_name+'.csv'), index=False)
    
def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for /// with /// Dataset.
        """
    )
    parser.add_argument("--config", type=str, help="path of a config file")

    parser.add_argument(
        "--ds_dir",
        required=False,
        default="../SummarizationDataset",
        help="dataset dir",
    )
    
    parser.add_argument(
        "--save_dir",
        required=False,
        default="../SummarizationDataset"
    )
    
    parser.add_argument(
        "--fps",
        required=False,
        default=30,
        help="dataset dir",
    )
    return parser.parse_args()



if __name__ == "__main__":
    main()

