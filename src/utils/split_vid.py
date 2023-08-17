import argparse
from tqdm import tqdm
import os
from pathlib import Path

def main():
    args = get_arguments()
    ds_dir = Path(args.ds_dir)
    save_dir = Path(args.save_dir)
    fps = args.fps
    
    input_path = ds_dir  / "videos"
    output_path = save_dir  / "video_split"
    
    videos_to_imgs(output_path=output_path,input_path=input_path,start_video_index=0, fps=fps)

def videos_to_imgs(output_path='',input_path='',pattern='*mp4',fps=30,start_video_index=0):
    dirs = list(input_path.glob(pattern))
    dirs = dirs[start_video_index:]
    os.makedirs(output_path, exist_ok=True)
    
    for i, vid_path in enumerate(tqdm(dirs)):
        file_name = vid_path.stem
        out_folder = output_path / file_name
        out_folder.mkdir(exist_ok=True)
        os.system(
            f'ffmpeg -i {vid_path} -vf "scale=250:250, fps=30" {out_folder/file_name}_%6d.png '
        )
        print("Done extractin: {}".format(i+1))

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

