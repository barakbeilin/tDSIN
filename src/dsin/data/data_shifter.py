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


def image_existance_report(path, csv_name, header: str = None, delimiter: str = '\n'):
    df = pd.read_csv(Path(path)/Path(csv_name),
                     header=None, delimiter=delimiter)
    missing_paths = []
    existing_paths = []
    for i in range(len(df)):
        img_path = Path(path)/df.iloc[i][0]
        if os.path.exists(img_path):
            existing_paths.append(str(img_path))
        else:
            missing_paths.append(str(img_path))
    return missing_paths, existing_paths, len(df)


if __name__ == "__main__":
    # copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_val.txt')
    # copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_train.txt')
    # copier(csv_path='/mnt/code/repos/tDSIN/src/dsin/data/nano_KITTI_stereo_test.txt')
    missing_paths, existing_paths, total = image_existance_report(
        path='/mnt/code/repos/tDSIN/src/dsin/data',
        csv_name='KITTI_stereo_val.txt')
    print(len(missing_paths))
    print(total)
    # print('\n'.join(['missing_paths', *missing_paths]))
    # input()
    # print('\n'.join(['existing_paths', *existing_paths]))
