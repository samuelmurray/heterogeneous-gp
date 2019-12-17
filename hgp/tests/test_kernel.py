import tensorflow as tf

from hgp.kernel import Kernel


class TestKernel(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Kernel("kernel")


if __name__ == "__main__":
    tf.test.main()
