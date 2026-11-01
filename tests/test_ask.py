import time

from meeting_assistant.ask import MAX_WORDS, relevant_excerpt
from meeting_assistant.transcript import Segment


def _segments(texts):
    now = time.time()
    return [Segment(now + i, now + i + 1, t) for i, t in enumerate(texts)]


def test_relevant_excerpt_keeps_matching_lines_with_neighbours_in_order():
    segs = _segments(
        ["Welcome everyone.", "Budget is twelve thousand.", "Unrelated chatter.", "More chatter.",
         "Even more chatter.", "Still chatter.", "The offsite is in Lisbon in December.",
         "Flights by end of October.", "Bye."]
    )
    out = relevant_excerpt(segs, "Where is the offsite and what is the budget?", max_words=100)
    lines = out.splitlines()
    assert any("Lisbon" in l for l in lines) and any("twelve thousand" in l for l in lines)
    assert "Welcome everyone." in out  # neighbour of a hit
    assert "…" in out  # gap marker between non-adjacent regions
    assert lines.index(next(l for l in lines if "twelve" in l)) < lines.index(next(l for l in lines if "Lisbon" in l))


def test_relevant_excerpt_respects_word_budget():
    segs = _segments([f"budget line number {i} with several extra words in it" for i in range(50)])
    out = relevant_excerpt(segs, "what is the budget?", max_words=40)
    words = [w for w in out.split() if w != "…" and not w.startswith("[")]  # ignore gap marks and timestamps
    assert 0 < len(words) <= 40


def test_max_words_fits_small_context():
    assert MAX_WORDS <= 2500
