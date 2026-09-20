"""Stage 2 guard against speaker attributions the summarizer invented (FINDINGS.md section 11)."""

import pytest

from hcv_sum.anchored_summarization import strip_invented_attribution

SOURCE = ("CHAIR POWELL. The unemployment rate at 4.2 percent is a very healthy unemployment rate. "
          "Housing services have behaved well. The staff report says inflation eased.")


@pytest.mark.parametrize("generated, expected, tag", [
    # Real DistilBART outputs on the FOMC press-conference transcript.
    ("The unemployment rate at 4.2 percent is a very healthy unemployment rate, President Obama says.",
     "The unemployment rate at 4.2 percent is a very healthy unemployment rate.", "President Obama says"),
    ("Housing services, housing services have behaved well, he adds.",
     "Housing services, housing services have behaved well.", "he adds"),
    ("Committee chairman says it's not too late to avoid a recession.",
     "It's not too late to avoid a recession.", "Committee chairman says"),
])
def test_invented_attribution_is_removed(generated, expected, tag):
    assert strip_invented_attribution(generated, SOURCE) == (expected, tag)


@pytest.mark.parametrize("generated", [
    "The staff report says inflation eased.",         # non-person subject: would become a bare assertion
    "The report says inflation eased.",
    "Inflation eased, the Chair said.",               # lower-case subject: left alone (conservative)
    "Housing services have behaved well.",            # no tag at all
])
def test_non_person_or_absent_tags_are_left_alone(generated):
    assert strip_invented_attribution(generated, SOURCE) == (generated, None)


def test_attribution_present_in_the_source_is_kept():
    source = "Powell says inflation eased a lot this year."
    assert strip_invented_attribution("Powell says inflation eased.", source) == ("Powell says inflation eased.", None)
