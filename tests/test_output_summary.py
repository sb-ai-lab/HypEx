# pytest -s -v tests/test_output_summary.py
import pytest

from hypex.ui.base import Output


class _Reporter:
    def __init__(self, value="summary-table"):
        self.value = value

    def report(self, experiment_data):
        return self.value


class _TwoTablesOutput(Output):
    summary: str
    extra: str


@pytest.fixture
def output():
    out = _TwoTablesOutput(
        summary_reporter=_Reporter("S"),
        additional_reporters={"extra": _Reporter("E")},
    )
    out.extract(object())
    return out


def test_summary_is_main_table(output):
    assert output.summary == "S"
    assert output.extra == "E"


def test_resume_alias_warns(output):
    with pytest.warns(DeprecationWarning, match="summary"):
        assert output.resume == "S"


def test_resume_setter_warns_and_sets_summary(output):
    with pytest.warns(DeprecationWarning):
        output.resume = "new"
    assert output.summary == "new"


def test_resume_reporter_kwarg_deprecated():
    reporter = _Reporter()
    with pytest.warns(DeprecationWarning, match="summary_reporter"):
        out = Output(resume_reporter=reporter)
    assert out.summary_reporter is reporter


def test_resume_reporter_attribute_deprecated(output):
    with pytest.warns(DeprecationWarning):
        assert output.resume_reporter is output.summary_reporter


def test_reporter_required():
    with pytest.raises(TypeError):
        Output()


def test_repr_is_full_report_summary_first(output):
    text = repr(output)
    assert text.startswith("_TwoTablesOutput:")
    assert text.index("summary:") < text.index("extra:")
    html = output._repr_html_()
    assert "summary" in html and "extra" in html


def test_repr_before_extract_does_not_fail():
    out = _TwoTablesOutput(summary_reporter=_Reporter())
    assert "no tables available" in repr(out)
