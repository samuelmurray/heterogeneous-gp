import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--testexamples", action="store_true", default=False, help="run tests for example scripts"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--testexamples"):
        # --testexamples given in cli: do not skip examples tests
        return
    skip_examples = pytest.mark.skip(reason="pass --testexamples option to run")
    for item in items:
        if "example" in item.keywords:
            item.add_marker(skip_examples)
