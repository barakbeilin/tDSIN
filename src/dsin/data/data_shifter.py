from pathlib import Path
import os
from shutil import copyfile
import pandas as pd


def copier(dst_root=Path("/mnt/code/repos/tDSIN/src/dsin/data/example"),
           src_root=Path("/mnt/code/repos/tDSIN/src/dsin/data"),
           csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_val.txt'):

    df = pd.read_csv(csv_path, header=None, delimiter='\n')
    for i in range(len(df)):
        file_path = Path(df.iloc[i][0])
        os.makedirs(dst_root/file_path.parent, exist_ok=True)
        copyfile(src_root/file_path, dst_root/file_path)


if __name__ == "__main__":
    copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_val.txt')
    copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_train.txt')
    copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_test.txt')