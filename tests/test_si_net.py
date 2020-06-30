import unittest
from dsin.ae.si_net import *
from tests import consts


class TestSiNet(unittest.TestCase):
    def setUp(self):
        self.net = SiNet()

    def test_enc_struct(self):
        self.assertEqual(consts.si_net_txt, repr(self.net))

    def test_sinet_flow(self):
        x = torch.randn([2, 3, 256, 256])
        y = self.net(x)
        self.assertEqual(tuple(y.shape), (2, 3, 256, 256))

    def test_init_weight_to_kaminig(self):

        lreul_gain = torch.sqrt(torch.tensor(2.0))  # roughly
        i = 0
        for param in self.net.parameters():
            if len(param.shape) == 4:
                # param = [channel_out, channel_in , KER_H , KER_L]
                # fan_in = CHANNEL_IN * KERNEL_SIZE = CHANNEL_IN * KER_H * KER_L
                fan_in = param.shape[1] * (param.shape[2] * param.shape[3])
                kaiming_uniform_bound = lreul_gain * torch.sqrt(
                    torch.tensor(3.0) / fan_in)
                
                calc_variance = (2 * kaiming_uniform_bound) ** 2 / 12
                self.assertAlmostEqual(torch.mean(param), 0.0, delta=0.1)
                print(i)
                i += 1
                self.assertAlmostEqual(torch.var(param), calc_variance, delta=0.05)


if __name__ == "__main__":
    unittest.main()
