general_paths = ["/mnt/code/repos/tDSIN/src/dsin/data/KITTI_general_train.txt",
                 "/mnt/code/repos/tDSIN/src/dsin/data/KITTI_general_val.txt",
                 "/mnt/code/repos/tDSIN/src/dsin/data/KITTI_general_test.txt",
                 ]

stereo_paths = ["/mnt/code/repos/tDSIN/src/dsin/data/KITTI_stereo_train.txt",
                "/mnt/code/repos/tDSIN/src/dsin/data/KITTI_stereo_val.txt",
                "/mnt/code/repos/tDSIN/src/dsin/data/KITTI_stereo_test.txt",
                ]
file_type = ["train", "val", "test"]

for general_path, stereo_path, file_type in zip(general_paths, stereo_paths, file_type):
    total_lines = []

    with open(general_path, 'r') as general_reader:
        general_lines = general_reader.readlines()
        total_lines.extend(general_lines)

    with open(stereo_path, 'r') as stereo_reader:
        stereo_lines = stereo_reader.readlines()
        total_lines.extend(stereo_lines)

    print(f"{len(total_lines)}")

    total_lines = list(set(total_lines))
    print(f"{len(total_lines)}")

    import ipdb; ipdb.set_trace()
    total_lines_with_mock_si_img = [total_lines[0]] * 2*  len(total_lines)
    # set real images in even index
    total_lines_with_mock_si_img[::2] = total_lines
    with open(f"/mnt/code/repos/tDSIN/src/dsin/data/KITTI_baseline_{file_type}.txt", 'w') as writer:
        writer.writelines(total_lines_with_mock_si_img)
