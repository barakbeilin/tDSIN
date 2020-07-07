import unittest
from dsin.ae.data_manager.data_loader import (
    SideinformationImageImageList, ImageSiTuple)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.image_list = SideinformationImageImageList.from_csv(
            path="/mnt/code/repos/tDSIN/src/dsin/data",
            csv_names=["tiny_KITTI_stereo_train.txt"])

    def test_si_image_list_work(self):
        self.assertEqual(len(self.image_list), 50)
        self.assertEqual(str(
            self.image_list[0].__class__),
            "<class 'dsin.ae.data_manager.data_loader.ImageSiTuple'>")
        self.assertEqual(tuple(self.image_list[0].img.shape), (3, 370, 1226))
        self.assertEqual(
            tuple(self.image_list[0].si_img.shape), (3, 370, 1226))


if __name__ == "__main__":
    unittest.main()
