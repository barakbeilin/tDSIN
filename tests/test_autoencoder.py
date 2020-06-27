import unittest
from dsin.ae.autoencoder_imgcomp import *
from tests import consts


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = Encoder.create_module_from_const()

    def test_enc_struct(self):

        self.assertEqual(consts.enc_txt, repr(self.enc))

    def test_encoder_flow(self):
        x = torch.rand([1, 3, 10, 10])
        y= self.enc(x)


class TestDecoder(unittest.TestCase):
    def test_dec_struct(self):

        dec = Decoder.create_module_from_const()

        self.assertEqual(consts.dec_txt, repr(dec))

    def test_decoder_flow(self):
        pass


if __name__ == "__main__":
    unittest.main()
