import pytest

from locus_eval.scenarios import build_suite


@pytest.fixture
def suite():
    return build_suite(17)


@pytest.fixture
def scenario(suite):
    return next(
        case
        for case in suite
        if case.faulty_component == "router" and case.components[0] == "router"
    )
