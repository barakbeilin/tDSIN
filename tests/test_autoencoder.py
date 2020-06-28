import unittest
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


class TestNormalization(unittest.TestCase):
    def test_norm_flow(self):
        normalize = ChangeImageStatsToKitti(ChangeState.NORMALIZE)
        denormalize = ChangeImageStatsToKitti(ChangeState.DENORMALIZE)
        unit = ChangeImageStatsToKitti(ChangeState.OFF)
        x = torch.randn([2, 3, 5, 5])

        self.assertAlmostEqual(torch.mean(x).data, 0.0, delta=0.2)
        self.assertAlmostEqual(torch.var(x).data, 1.0, delta=0.3)

        denorm = denormalize(x)
        self.assertTrue(tuple(denorm.shape) == (2, 3, 5, 5))
        self.assertTrue(
            torch.allclose(
                torch.mean(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [93.70454143384742, 98.28243432206516, 94.84678088809876],
                    dtype=torch.float32,
                ),
                atol=30,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [5411.79935676, 5758.60456747, 5890.31451232], dtype=torch.float32,
                ),
                atol=0,
                rtol=0.5,
            )
        )

        normalized = normalize(denorm)

        self.assertTrue(
            torch.allclose(
                torch.mean(normalized, dim=(0, 2, 3)),
                torch.tensor([0, 0, 0], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(normalized, dim=(0, 2, 3)),
                torch.tensor([1, 1, 1], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(tuple(normalized.shape) == (2, 3, 5, 5))


if __name__ == "__main__":
    unittest.main()
