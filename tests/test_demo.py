from __future__ import annotations

import pytest

from rlcd import demo


def test_choice_request_lists_every_option():
    state, schema = demo.build_request("s", "q?", "choice", ["a", "b", "c"])
    assert state == {"state": {"text": "s"}}
    assert schema["answer"] == {
        "type": "choice",
        "instructions": "q?",
        "criteria": {"a": None, "b": None, "c": None},
    }


def test_score_request_keeps_the_level_order_and_noul_needs_no_options():
    _, schema = demo.build_request("s", "q?", "score", ["calm", "angry"])
    assert schema["answer"]["criteria"] == ["calm", "angry"]
    _, schema = demo.build_request("s", "q?", "noul", None)
    assert schema["answer"] == {"type": "noul", "instructions": "q?"}


def test_bad_requests_are_rejected():
    with pytest.raises(ValueError):
        demo.build_request("s", "q", "choice", ["only one"])
    with pytest.raises(ValueError):
        demo.build_request("s", "q", "ranking", ["a", "b"])
