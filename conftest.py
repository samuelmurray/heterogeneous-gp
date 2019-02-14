import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runexamples", action="store_true", default=False, help="run tests for examples"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runexamples"):
        # --runexamples given in cli: do not skip examples tests
        return
    skipt_examples = pytest.mark.skip(reason="need --runexaples option to run")
    for item in items:
        if "examples" in item.keywords:
            item.add_marker(skipt_examples)
