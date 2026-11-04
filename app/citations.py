"""Reading a citation the model wrote into provision IDs from the index.

Used by the citation guard (app.answer.guard) and the eval's grader (eval.run), so both read a citation
the same way (session 15). The first real run showed the model writing forms a one-ID-per-string rule
can't read:
    packed      "1026.13(c)(1), § 1026.13(c)(2)"     -> 1026.13(c)(1), 1026.13(c)(2)
    sub-point   "1005.12(a)-5.iv.B"                  -> 1005.12(a)-5   (sub-points are part of comment 5)
    range       "1005.11(c)(2)(iii)-(iv)"            -> 1005.11(c)(2)(iii), 1005.11(c)(2)(iv)
    in a phrase "see 1026.9(c)(2)(i) and the notice"  -> 1026.9(c)(2)(i)

References are read with the search's own reader (app.retrieve.read_refs and resolve). What can't be
turned into an index ID is sorted, not silently dropped:
    too_broad    names something real but with no text of its own: a whole section ("1005.11") or a
                 paragraph heading, so it doesn't say which provision supports the answer
    not_found    shaped like a reference, but nothing in the index has that number ("1026.99(z)")
    no_part      a short form without its part ("comment 12(b)-1") that exists in both Reg E and Reg Z;
                 the prompt asks for the full part, so it isn't guessed
    unreadable   the string holds no reference at all ("visa-rules")
"""
import re
from dataclasses import dataclass, field

from app import retrieve as rt

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
         "xvi", "xvii", "xviii", "xix", "xx"]
# "1005.11(c)(2)(iii)-(iv)": an ID, then a dash and one more label at the same level as its last label
RANGE = re.compile(r"\b(10(?:05|26)\.\d{1,2}(?:\(\s?[0-9A-Za-z]{1,5}\s?\))*)\(\s?([0-9A-Za-z]{1,5})\s?\)"
                   r"\s?(?:[-‐-―]|\bthrough\b|\bto\b)\s?\(\s?([0-9A-Za-z]{1,5})\s?\)(?!\s?\()")
# the end is one label: "(c)(1) through (c)(2)" is not a range of single labels, and only (c)(1) is read
MAX_RANGE = 12            # a longer range is read as its two ends only


@dataclass
class Reading:
    """What one citation string names. `ids` are exact index IDs in the index's spelling."""
    written: str
    ids: list = field(default_factory=list)
    too_broad: list = field(default_factory=list)
    not_found: list = field(default_factory=list)
    no_part: list = field(default_factory=list)
    unreadable: bool = False

    @property
    def problems(self):
        """Everything that isn't an exact index ID, as written or as read, for flags and counts."""
        return self.too_broad + self.not_found + self.no_part + ([self.written] if self.unreadable else [])


def _sequence(start, end, exists):
    """Labels from start to end at one level, or just the two ends if the kind is unclear or it's long.
    `exists(label)` says whether prefix(label) is in the index, to tell roman (i)-(v) from letters."""
    def roman_ok(a, b):
        return a in ROMAN and b in ROMAN and ROMAN.index(a) < ROMAN.index(b)
    if start.isdigit() and end.isdigit() and int(start) < int(end):
        seq = [str(n) for n in range(int(start), int(end) + 1)]
    elif roman_ok(start, end) and (len(start) > 1 or len(end) > 1 or exists("ii")):
        seq = ROMAN[ROMAN.index(start):ROMAN.index(end) + 1]
    elif len(start) == len(end) == 1 and start.isalpha() and end.isalpha() and start.islower() == end.islower() \
            and start < end:
        seq = [chr(c) for c in range(ord(start), ord(end) + 1)]
    else:
        return [start, end]
    return seq if len(seq) <= MAX_RANGE else [start, end]


def expand_ranges(text, lower_ids):
    """Write each range out as a list: "1005.11(c)(2)(iii)-(iv)" -> "1005.11(c)(2)(iii), 1005.11(c)(2)(iv)"."""
    def one(m):
        prefix = re.sub(r"\s", "", m.group(1))
        start, end = m.group(2), m.group(3)
        exists = lambda label: f"{prefix}({label})".lower() in lower_ids
        return ", ".join(f"{prefix}({label})" for label in _sequence(start, end, exists))
    return RANGE.sub(one, text)


def read_citation(written, provisions):
    """Read one citation string. `provisions` maps lower-case ID -> the index's spelling."""
    r = Reading(written=written)
    text = expand_ranges(written, provisions)
    refs = rt.read_refs(text)
    if not refs:
        r.unreadable = True
        return r
    for ref in refs:
        found, missing = rt.resolve([ref], provisions.values())
        if missing:
            r.not_found.append(ref[0])
        elif len(found) > 1:
            r.no_part.append(ref[0])
        elif found[0].lower() in provisions:
            if provisions[found[0].lower()] not in r.ids:
                r.ids.append(provisions[found[0].lower()])
        else:
            r.too_broad.append(found[0])
    return r


def read_all(strings, provisions):
    """Read several citation strings: (IDs in order of first mention, list of Readings)."""
    readings = [read_citation(s, provisions) for s in strings]
    ids = list(dict.fromkeys(i for rd in readings for i in rd.ids))
    return ids, readings


def marker(provision_id):
    """How the prompt writes an inline citation: [§ 1005.6(b)(1)] or [comment 1005.6(b)-1]."""
    is_comment = "-" in provision_id.split(".", 1)[1]
    return f"comment {provision_id}" if is_comment else f"§ {provision_id}"
