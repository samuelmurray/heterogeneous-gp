import pytest
import tensorflow as tf


@pytest.mark.examples
class TestImportRun(tf.test.TestCase):
    def setUp(self):
        pass

    def test_run_gp(self):
        try:
            from tfgp.examples import run_gp
        except Exception as e:
            self.fail(f"'import run_gp' raised exception: {e}")

    def test_run_gplvm(self):
        try:
            from tfgp.examples import run_gplvm
        except Exception as e:
            self.fail(f"'import run_gplvm' raised exception: {e}")

    def test_run_imputation_classification(self):
        try:
            from tfgp.examples import run_imputation_classification
        except Exception as e:
            self.fail(f"'import run_imputation_classification' raised exception: {e}")

    def test_run_mlgp(self):
        try:
            from tfgp.examples import run_mlgp
        except Exception as e:
            self.fail(f"'import run_mlgp' raised exception: {e}")

    def test_run_mlgplvm(self):
        try:
            from tfgp.examples import run_mlgplvm
        except Exception as e:
            self.fail(f"'import run_mlgplvm' raised exception: {e}")


if __name__ == "__main__":
    tf.test.main()
