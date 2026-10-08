"""Ask a model a typed question and print a probability for every option.

    python -m rlcd.demo --model MERGED_DIR --state "Ticket: charged twice for order A-104, wants a refund." \\
        --question "Which team should handle this?" --type choice --options returns,billing,other
    python -m rlcd.demo --model MERGED_DIR --state "..." --question "Does the customer ask for a refund?" --type noul
    python -m rlcd.demo --model MERGED_DIR --state "..." --question "How frustrated is the customer?" \\
        --type score --options calm,frustrated,"very frustrated"

``MERGED_DIR`` is a decider checkpoint, for example an Exp A adapter merged into the start model with
``scripts/merge_expA_adapter.py``. Needs one CUDA GPU and decider (https://github.com/Mapika/decider) installed. The
answer is read in one forward pass per question (``SCORE``: one pass per level, normalised): the distribution
you see is the object the experiments train and score.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def build_request(
    state: str, question: str, qtype: str, options: list[str] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``system_one`` (state, schema) pair for one typed question."""
    if qtype not in ("noul", "choice", "score"):
        raise ValueError("type must be noul, choice or score")
    spec: dict[str, Any] = {"type": qtype, "instructions": question}
    if qtype == "choice":
        if not options or len(options) < 2:
            raise ValueError("a choice question needs at least two --options")
        spec["criteria"] = {o: None for o in options}
    elif qtype == "score":
        if not options or len(options) < 2:
            raise ValueError("a score question needs at least two ordered --options")
        spec["criteria"] = list(options)
    return {"state": {"text": state}}, {"answer": spec}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", required=True, help="a decider checkpoint directory or Hugging Face id"
    )
    ap.add_argument("--state", required=True, help="the situation, as text")
    ap.add_argument("--question", required=True)
    ap.add_argument("--type", dest="qtype", choices=["noul", "choice", "score"], default="choice")
    ap.add_argument(
        "--options", default="", help="comma-separated options (choice) or ordered levels (score)"
    )
    ap.add_argument("--json", action="store_true", help="print the raw response")
    a = ap.parse_args()
    options = [o.strip() for o in a.options.split(",") if o.strip()] or None
    state, schema = build_request(a.state, a.question, a.qtype, options)
    from decider.infer import Decider  # needs a GPU; imported late so --help and the tests do not

    out = Decider(a.model).system_one(state, schema)["answers"]["answer"]
    if a.json:
        print(json.dumps(out, indent=2))
        return
    probs = out.get("probabilities") or {
        "yes": out.get("noul"),
        "no": None if out.get("noul") is None else 1 - out["noul"],
    }
    width = max(len(str(k)) for k in probs)
    for k, v in sorted(probs.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"{str(k):<{width}}  {v:.3f}  {'#' * round(40 * v)}")


if __name__ == "__main__":
    main()
