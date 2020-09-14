import unittest
import torch
from dsin.ae.autoencoder_imgcomp import *
from tests import consts


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = Encoder.create_module_from_const()

    def test_enc_struct(self):
        self.assertEqual(consts.enc_txt, repr(self.enc))

    def test_encoder_flow(self):
        x = torch.randn([2, 3, 128, 128])
        y = self.enc(x)
        self.assertEqual(tuple(y.shape), (2, 33, 16, 16))


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.dec = Decoder.create_module_from_const()

    def test_dec_struct(self):
        self.assertEqual(consts.dec_txt, repr(self.dec))

    def test_decoder_flow(self):
        x = torch.randn([2, 32, 16, 16])
        y = self.dec(x)
        self.assertEqual(tuple(y.shape), (2, 3, 128, 128))


if __name__ == "__main__":
    unittest.main()
