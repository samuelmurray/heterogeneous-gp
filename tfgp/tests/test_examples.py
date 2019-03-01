import pytest
import tensorflow as tf


@pytest.mark.example
class TestImportRun(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_run_batch_mlgp(self) -> None:
        try:
            from tfgp.examples import run_batch_mlgp
        except Exception as e:
            self.fail(f"'import run_batch_mlgp' raised exception: {e}")

    def test_run_batch_mlgplvm(self) -> None:
        try:
            from tfgp.examples import run_batch_mlgplvm
        except Exception as e:
            self.fail(f"'import run_batch_mlgplvm' raised exception: {e}")

    def test_run_gp(self) -> None:
        try:
            from tfgp.examples import run_gp
        except Exception as e:
            self.fail(f"'import run_gp' raised exception: {e}")

    def test_run_gplvm(self) -> None:
        try:
            from tfgp.examples import run_gplvm
        except Exception as e:
            self.fail(f"'import run_gplvm' raised exception: {e}")

    def test_run_imputation_classification(self) -> None:
        try:
            from tfgp.examples import run_imputation_classification
        except Exception as e:
            self.fail(f"'import run_imputation_classification' raised exception: {e}")

    def test_run_mlgp(self) -> None:
        try:
            from tfgp.examples import run_mlgp
        except Exception as e:
            self.fail(f"'import run_mlgp' raised exception: {e}")

    def test_run_mlgplvm(self) -> None:
        try:
            from tfgp.examples import run_mlgplvm
        except Exception as e:
            self.fail(f"'import run_mlgplvm' raised exception: {e}")

    def test_run_vae_mlgplvm(self) -> None:
        try:
            from tfgp.examples import run_vae_mlgplvm
        except Exception as e:
            self.fail(f"'import run_vae_mlgplvm' raised exception: {e}")


if __name__ == "__main__":
    tf.test.main()
