import unittest
from dsin.ae.data_manager.data_loader import (
    SideinformationImageImageList, ImageSiTuple)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.image_list = SideinformationImageImageList.from_csv(
            path="/mnt/code/repos/tDSIN/src/dsin/data",
            csv_names=["tiny_KITTI_stereo_train.txt",
                       "tiny_KITTI_stereo_val.txt"])

    def test_image_list_work(self):
        self.assertEqual(len(self.image_list), 80)
        self.assertEqual(str(
            self.image_list[0].__class__),
            "<class 'dsin.ae.data_manager.data_loader.ImageSiTuple'>")
        self.assertEqual(tuple(self.image_list[0].img.shape), (3, 370, 1226))
        self.assertEqual(
            tuple(self.image_list[0].si_img.shape), (3, 370, 1226))

    def test_data_bunch_creation(self):
        batchsize = 1
        data = (self.image_list
                .split_by_valid_func(lambda x: 'testing' in x)
                .label_from_func(lambda x: x)
                .databunch(bs=batchsize))
        self.assertEqual(len(data.train_ds), 50)
        self.assertEqual(len(data.valid_ds), 30)


if __name__ == "__main__":
    unittest.main()
