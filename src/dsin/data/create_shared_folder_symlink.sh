#!/bin/bash
sudo vmhgfs-fuse .host:/ /mnt/hgfs/ -o allow_other -o uid=1000
ln -s /mnt/hgfs/deepl/data_stereo_flow_multiview/training/ /mnt/code/repos/tDSIN/src/dsin/data/data_stereo_flow_multiview/training
ln -s /mnt/hgfs/deepl/data_stereo_flow_multiview/testing/ /mnt/code/repos/tDSIN/src/dsin/data/data_scene_flow_multiview/testing
