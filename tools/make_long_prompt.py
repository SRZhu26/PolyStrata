"""Build a prompt of an exact token count, for both engines to read the same text.

The point of a head-to-head is that both engines process the SAME tokens.  A prompt built from a
token count in one tokenizer and re-tokenized by the other only stays comparable if the two agree, so
the count is checked against the other engine's tokenizer (llama-tokenize) before any speed is measured.

    python tools/make_long_prompt.py --gguf MODEL.gguf --tokens 130816 --out prompt.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strata_tokenizer as ST

# Prose rather than one repeated line: a repeated line makes every token's KV row the same, which
# flatters any cache that keys on content.  The paragraphs vary in length and punctuation, as a real
# long context does.
PARAGRAPHS = [
    "The lighthouse keeper had kept the light for thirty-one years, and in that time he had learned the "
    "difference between a storm that would pass and one that would not. The passing ones announced themselves "
    "with a certain impatience, a drumming on the glass that rose and fell like an argument. The others came in "
    "silence, and the silence was worse, because it meant the sea had decided something and was not going to say what.",
    "Every morning he climbed the ninety-four steps with a bucket of paraffin and a rag, and every morning he "
    "counted the steps aloud, not because he had forgotten the number but because counting was a way of proving to "
    "himself that the morning had actually happened. The logbook recorded the weather, the shipping, and the state of the "
    "lamp, and on the days when nothing had happened the logbook recorded that too.",
    "There is a particular kind of loneliness that comes from being responsible for something that other people depend on "
    "without ever seeing. The keeper did not think of it as loneliness. He thought of it as arithmetic: one lamp, one man, "
    "one night, and the arithmetic worked out or it did not, and if it did not then a ship went onto the rocks and the "
    "question of whose fault it was would be settled by a court in a town he had never visited.",
    "When the supply boat came in April it brought flour, salt, lamp oil, and a bundle of newspapers that were three "
    "weeks old. He read them in order, slowly, and by the time he reached the end he was reading about a world that had "
    "already moved on. The boat also brought letters, and the letters were worse than the newspapers, because the "
    "newspapers were about strangers and the letters were not.",
    "In the winter the sea came up the rock in a way that made the tower ring like a bell struck once and then not "
    "again for a long time. He had been told, when he took the post, that the tower had been built to withstand anything, "
    "and he had believed it, and he still believed it, but believing a thing and listening to it are not the same, and in "
    "the winter he did both.",
    "The lamp itself was a thing of glass and brass and a mechanism he could take apart and put together in the dark, "
    "which he had done once, in the dark, when the mechanism had jammed at two in the morning and the light had gone "
    "out and a boat had been in the water below. He had put it together in eleven minutes and he had never told anyone "
    "how frightened he had been, because telling it would have made it real.",
    "There was a girl in the town who came up the rock once a year with her father, who was a fisherman, and she had "
    "asked him what he did all day and he had said that he kept the light, and she had said that was not what she had "
    "asked, and he had thought about it for a long time afterward and had decided that she was right, and that he did not "
    "know what he did all day either.",
]


def build(tok, target: int) -> str:
    """A text of exactly `target` tokens under `tok`."""
    block = "\n\n".join(PARAGRAPHS) + "\n\n"
    n = len(tok.encode(block, parse_special=True))
    reps = target // n + 1
    text = block * reps
    ids = tok.encode(text, parse_special=True)
    # Cut at a token boundary by taking the ids and decoding them back: decode() of a prefix is the
    # text of that prefix, and re-encoding it can merge across the cut, so the count is re-checked and
    # the cut walked back until it is exact.
    ids = ids[:target]
    for back in range(0, 512):
        cand = ids[:len(ids) - back] if back else ids
        t = tok.decode(cand)
        got = len(tok.encode(t, parse_special=True))
        if got == len(cand):
            return t
    raise RuntimeError("no exact token cut found within 512 tokens of the target")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", type=int, default=130816)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = ST.Tokenizer.from_gguf(a.gguf)
    text = build(tok, a.tokens)
    n = len(tok.encode(text, parse_special=True))
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print("tokens=%d chars=%d bytes=%d out=%s" % (n, len(text), len(text.encode()), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
