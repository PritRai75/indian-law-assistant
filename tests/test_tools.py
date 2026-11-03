"""Tests for app/tools.py (`dispute_timeline`) and data/us_federal_holidays.json.

Every expected date here was worked out by hand from a calendar, never from the tool's output
(spec labelling rule 3). November 2026: Mon 2, Wed 11 Veterans Day (closed), Thu 26 Thanksgiving
(closed). July 2026: Sat 4 Independence Day (Friday 3 open). July 2027: Sun 4 (Monday 5 closed).
"""

import json
from datetime import date, timedelta

import pytest

from app import tools as T
from app.tools import dispute_timeline as dt


def dates(result):
    return {d["rule"]: d["date"] for d in result["deadlines"]}


def by_what(result, start):
    return next(d for d in result["deadlines"] if d["what"].startswith(start))


# ------------------------------------------------------------------ holiday table

FED_TABLE = {   # Federal Reserve holiday schedule as published (frbservices.org, federalreserve.gov), read 3 Nov 2026
    2026: ["01-01", "01-19", "02-16", "05-25", "06-19", "07-04", "09-07", "10-12", "11-11", "11-26", "12-25"],
    2027: ["01-01", "01-18", "02-15", "05-31", "06-19", "07-04", "09-06", "10-11", "11-11", "11-25", "12-25"],
}


def nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def last_weekday(year, month, weekday):
    d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def rule_holidays(year):
    """The eleven holidays by their fixed-date and nth-weekday rules."""
    return [date(year, 1, 1), nth_weekday(year, 1, 0, 3), nth_weekday(year, 2, 0, 3), last_weekday(year, 5, 0),
            date(year, 6, 19), date(year, 7, 4), nth_weekday(year, 9, 0, 1), nth_weekday(year, 10, 0, 2),
            date(year, 11, 11), nth_weekday(year, 11, 3, 4), date(year, 12, 25)]


def test_rules_reproduce_the_published_fed_table():
    for year, days in FED_TABLE.items():
        assert [d.strftime("%m-%d") for d in rule_holidays(year)] == days


def test_holiday_file_matches_the_rules_and_weekend_rule():
    with open(T.HOLIDAYS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    want = []
    for year in (2025, 2026, 2027):
        for d in rule_holidays(year):
            if d.weekday() == 6:                 # Sunday: closed the following Monday
                want.append(d + timedelta(days=1))
            elif d.weekday() < 5:                # Saturday: no weekday closure (open the Friday before)
                want.append(d)
    assert [h["date"] for h in data["closed"]] == [d.isoformat() for d in want]
    assert data["covers"] == ["2025-01-01", "2027-12-31"]


# ------------------------------------------------------------------ Reg E unauthorized, 1005.6(b)

def test_comment_6b1_1_example():
    # Learns of the theft on Wednesday; notice by midnight Friday keeps the $50 limit.
    r = dt("E", "unauthorized", "2026-11-06", learned_of_loss="2026-11-04")
    assert dates(r)["1005.6(b)(1)"] == "2026-11-06"
    assert r["liability_tiers"] == ["max_50"] and r["notice_timely"] is True


def test_comment_6b2_1_example():
    # Learns Monday; the two business days end Wednesday; reported Friday, so the $500 tier.
    r = dt("E", "unauthorized", "2026-11-06", learned_of_loss="2026-11-02")
    assert dates(r)["1005.6(b)(1)"] == "2026-11-04"
    assert r["liability_tiers"] == ["max_500"] and r["notice_timely"] is False
    assert any("1005.6(b)(4)" in n for n in r["notes"])        # extenuating circumstances noted when late


def test_learning_day_and_weekend_not_counted():
    r = dt("E", "unauthorized", "2026-11-17", learned_of_loss="2026-11-13")   # Friday -> Mon 16, Tue 17
    assert dates(r)["1005.6(b)(1)"] == "2026-11-17" and r["liability_tiers"] == ["max_50"]


def test_weekday_holiday_skipped():
    r = dt("E", "unauthorized", "2026-11-12", learned_of_loss="2026-11-09")   # Tue 10, (Wed 11), Thu 12
    assert dates(r)["1005.6(b)(1)"] == "2026-11-12" and r["liability_tiers"] == ["max_50"]
    r = dt("E", "unauthorized", "2026-11-30", learned_of_loss="2026-11-25")   # (Thu 26), Fri 27, Mon 30
    assert dates(r)["1005.6(b)(1)"] == "2026-11-30"


def test_sunday_holiday_closes_monday():
    r = dt("E", "unauthorized", "2027-07-07", learned_of_loss="2027-07-02")   # (Mon 5), Tue 6, Wed 7
    assert dates(r)["1005.6(b)(1)"] == "2027-07-07"


def test_saturday_holiday_leaves_friday_open():
    r = dt("E", "unauthorized", "2026-07-06", learned_of_loss="2026-07-02")   # Fri 3, Mon 6
    assert dates(r)["1005.6(b)(1)"] == "2026-07-06"


def test_statement_rule_adds_a_tier():
    on_time = dt("E", "unauthorized", "2026-11-06", learned_of_loss="2026-11-02", statement_sent="2026-09-30")
    assert dates(on_time)["1005.6(b)(3)"] == "2026-11-29"                      # Sep 30 + 60 days
    assert on_time["liability_tiers"] == ["max_500"]
    late = dt("E", "unauthorized", "2026-11-06", learned_of_loss="2026-11-02", statement_sent="2026-08-31")
    assert dates(late)["1005.6(b)(3)"] == "2026-10-30"                         # Aug 31 + 60 days
    assert late["liability_tiers"] == ["max_500", "later_transfers_unprotected"]
    assert any("after 2026-10-30" in n for n in late["notes"])


def test_no_access_device():
    r = dt("E", "unauthorized", "2026-11-29", statement_sent="2026-09-30")    # day 60 itself is in time
    assert r["liability_tiers"] == ["none"] and r["notice_timely"] is True
    r = dt("E", "unauthorized", "2026-11-30", statement_sent="2026-09-30")    # day 61
    assert r["liability_tiers"] == ["later_transfers_unprotected"] and r["notice_timely"] is False
    assert any("comment 6(b)(3)-2" in n for n in r["notes"])


def test_reg_e_unauthorized_needs_a_date():
    assert dt("E", "unauthorized", "2026-11-06")["error"]["argument"] == "learned_of_loss"


# ------------------------------------------------------------------ Reg E errors, 1005.11

def test_standard_error_periods():
    r = dt("E", "error", "2026-11-02", statement_sent="2026-09-30", transfer_kind="standard")
    assert r["notice_timely"] is True
    assert by_what(r, "Notice of error")["date"] == "2026-11-29"
    assert by_what(r, "Determine")["date"] == "2026-11-17"     # 10 business days, Veterans Day skipped
    assert by_what(r, "Determine")["rule"] == "1005.11(c)(1)"
    assert by_what(r, "Or, to take longer")["date"] == "2026-11-17"
    assert by_what(r, "If the institution requires written")["date"] == "2026-11-17"
    assert by_what(r, "Finish")["date"] == "2026-12-17"        # Nov 2 + 45 days
    assert by_what(r, "Finish")["rule"] == "1005.11(c)(2)"


@pytest.mark.parametrize("kind,rule_note", [("pos_debit", "(c)(3)(ii)(B)"), ("foreign", "(c)(3)(ii)(A)")])
def test_ninety_days_for_pos_and_foreign(kind, rule_note):
    r = dt("E", "error", "2026-11-02", transfer_kind=kind)
    assert by_what(r, "Finish")["date"] == "2027-01-31"        # Nov 2 + 90 days
    assert by_what(r, "Finish")["rule"] == "1005.11(c)(3)(ii)"
    assert by_what(r, "Determine")["date"] == "2026-11-17"     # the 10-business-day period is unchanged
    assert any(rule_note in n for n in r["notes"])
    assert r["notice_timely"] is None                          # no statement date given


def test_new_account_periods():
    r = dt("E", "error", "2026-11-02", first_deposit="2026-10-20", transfer_date="2026-11-01",
           transfer_kind="standard")
    assert by_what(r, "Determine")["date"] == "2026-12-02"     # 20 business days: Veterans Day, Thanksgiving skipped
    assert by_what(r, "Determine")["rule"] == "1005.11(c)(3)(i)"
    assert by_what(r, "Finish")["date"] == "2027-01-31"
    assert by_what(r, "If the institution requires written")["date"] == "2026-11-17"   # stays 10


def test_new_account_boundary():
    # Oct 20 + 30 days = Nov 19: a transfer that day is still within 30 days after the first deposit.
    on = dt("E", "error", "2026-11-23", first_deposit="2026-10-20", transfer_date="2026-11-19", transfer_kind="standard")
    off = dt("E", "error", "2026-11-23", first_deposit="2026-10-20", transfer_date="2026-11-20", transfer_kind="standard")
    assert "20 business" in by_what(on, "Determine")["what"]
    assert "10 business" in by_what(off, "Determine")["what"]


def test_late_reg_e_error_notice():
    r = dt("E", "error", "2026-11-02", statement_sent="2026-08-31", transfer_kind="standard")
    assert r["notice_timely"] is False and len(r["deadlines"]) == 1
    assert any("comment 11(b)(1)-7" in n for n in r["notes"])


@pytest.mark.parametrize("kwargs,argument", [
    ({}, "transfer_kind"),
    ({"transfer_kind": "atm"}, "transfer_kind"),
    ({"transfer_kind": "standard", "first_deposit": "2026-10-20"}, "transfer_date"),
    ({"transfer_kind": "standard", "first_deposit": "2026-10-20", "transfer_date": "2026-10-01"}, "transfer_date"),
    ({"transfer_kind": "standard", "learned_of_loss": "2026-11-01"}, "learned_of_loss"),
    ({"transfer_kind": "standard", "cycle_closing_dates": ["2026-11-15"]}, "cycle_closing_dates"),
])
def test_reg_e_error_input_errors(kwargs, argument):
    assert dt("E", "error", "2026-11-02", **kwargs)["error"]["argument"] == argument


# ------------------------------------------------------------------ Reg Z billing errors, 1026.13

MONTHLY_15TH = ["2026-10-15", "2026-11-15", "2026-12-15", "2027-01-15", "2027-02-15"]


def test_billing_error_two_cycles_mid_cycle():
    # Received Nov 2, mid-cycle: the rest of that cycle plus the next two -> Jan 15 (comment 13(c)(2)-1).
    r = dt("Z", "error", "2026-11-02", statement_sent="2026-09-15", cycle_closing_dates=MONTHLY_15TH)
    assert r["notice_timely"] is True
    assert dates(r) == {"1026.13(b)(1)": "2026-11-14", "1026.13(c)(1)": "2026-12-02", "1026.13(c)(2)": "2027-01-15"}


def test_billing_error_received_on_closing_date():
    r = dt("Z", "error", "2026-11-15", cycle_closing_dates=MONTHLY_15TH)
    assert dates(r)["1026.13(c)(2)"] == "2027-01-15"            # cycles Nov 16-Dec 15 and Dec 16-Jan 15


def test_billing_error_ninety_day_cap_binds():
    r = dt("Z", "error", "2026-11-01", cycle_closing_dates=["2026-10-31", "2026-11-30", "2026-12-31", "2027-01-31"])
    assert dates(r)["1026.13(c)(2)"] == "2027-01-30"            # Nov 1 + 90 days, before Jan 31


def test_billing_error_without_cycles_gives_the_cap():
    r = dt("Z", "error", "2026-11-02")
    assert dates(r)["1026.13(c)(2)"] == "2027-01-31"
    assert any("90-day" in n for n in r["notes"])


def test_late_billing_error_notice():
    r = dt("Z", "error", "2026-11-02", statement_sent="2026-08-31")
    assert r["notice_timely"] is False and dates(r) == {"1026.13(b)(1)": "2026-10-30"}
    assert any("comment 12(b)(3)-3" in n for n in r["notes"])


def test_closing_dates_must_be_in_order():
    r = dt("Z", "error", "2026-11-02", cycle_closing_dates=["2026-12-15", "2026-11-15"])
    assert r["error"]["argument"] == "cycle_closing_dates"


# ------------------------------------------------------------------ Reg Z unauthorized use, 1026.12(b)

def test_reg_z_unauthorized_cap_and_conditions():
    r = dt("Z", "unauthorized", "2026-11-02")
    assert r["liability_tiers"] == ["max_50"] and r["deadlines"] == [] and r["notice_timely"] is None
    assert any("1026.12(b)(2)" in n for n in r["notes"])


@pytest.mark.parametrize("kwargs,argument", [
    ({"learned_of_loss": "2026-11-01"}, "learned_of_loss"),
    ({"statement_sent": "2026-10-01"}, "statement_sent"),
    ({"transfer_kind": "pos_debit"}, "transfer_kind"),
])
def test_reg_z_unauthorized_input_errors(kwargs, argument):
    assert dt("Z", "unauthorized", "2026-11-02", **kwargs)["error"]["argument"] == argument


# ------------------------------------------------------------------ general input handling

@pytest.mark.parametrize("args,argument", [
    (("X", "error", "2026-11-02"), "regulation"),
    (("E", "dispute", "2026-11-02"), "event"),
    (("Z", "error", None), "notice_date"),
    (("Z", "error", "11/02/2026"), "notice_date"),
])
def test_bad_core_arguments(args, argument):
    assert dt(*args)["error"]["argument"] == argument


def test_dates_outside_the_holiday_table():
    r = dt("E", "unauthorized", "2027-12-31", learned_of_loss="2027-12-30")   # counting runs into 2028
    assert r["error"]["argument"] == "learned_of_loss" and "2028-01-01 is outside" in r["error"]["message"]
    assert dt("E", "error", "2024-12-30", transfer_kind="standard")["error"]["argument"] == "notice_date"


def test_date_objects_accepted_and_assumptions_stated():
    r = dt("Z", "error", date(2026, 11, 2))
    assert r["notice_date"] == "2026-11-02"
    assert r["assumptions"] == [T.BUSINESS_DAY_ASSUMPTION, T.DAY_COUNT]


def test_tool_spec_matches_the_function():
    import inspect
    props = T.TOOL_SPEC["toolSpec"]["inputSchema"]["json"]["properties"]
    assert list(props) == list(inspect.signature(T.dispute_timeline).parameters)
    assert T.TOOL_SPEC["toolSpec"]["inputSchema"]["json"]["required"] == ["regulation", "event", "notice_date"]
    json.dumps(T.TOOL_SPEC)                                     # serialisable for the Converse API


def test_every_cited_paragraph_and_comment_exists():
    """Each regulation reference written in tools.py is a real provision in the corpus."""
    import re
    from ingest.chunk import load_provisions
    ids = {p["id"] for p in load_provisions()}
    src = open(T.__file__, encoding="utf-8").read()
    refs = set(re.findall(r"\b(10(?:05|26)\.\d+(?:\([a-zA-Z0-9]+\))*)", src))
    assert refs and all(r in ids or any(i.startswith(r + "(") for i in ids) for r in refs), \
        sorted(r for r in refs if r not in ids and not any(i.startswith(r + "(") for i in ids))
    comments = set(re.findall(r"comment (\d+(?:\([a-z0-9]+\))+-\d+)", src))
    assert comments and all(f"1005.{c}" in ids or f"1026.{c}" in ids for c in comments), comments
