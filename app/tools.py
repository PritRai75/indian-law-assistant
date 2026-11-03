"""The `dispute_timeline` tool: dispute deadlines and liability caps worked out from dates.

Plain Python, no model inside. A model is unreliable at counting business days in prose, so the
answer flow calls this instead (prompt rule 2), and experiment 3 measures what it adds.

Every rule below was checked against the corpus text (as of 2026-10-23) in session 11:

  Reg E, unauthorized transfers (1005.6(b))
    - Notice within 2 business days after learning of the loss or theft of the access device:
      at most the lesser of $50 or the transfers before notice                       (b)(1)
    - Later: at most the lesser of $500 or the sum defined in (b)(2)                     (b)(2)
    - Transfers on a statement not reported within 60 days of its transmittal: liability for
      transfers after those 60 days that timely notice would have prevented, on top of any
      tier above                                                                          (b)(3), comment 6(b)(3)-1
    - No lost or stolen access device: the $50 and $500 tiers don't apply                comment 6(b)(3)-2
    - Extenuating circumstances extend the limits to a reasonable period (never computed) (b)(4)
  Reg E, errors (1005.11)
    - Notice received no later than 60 days after the statement first showing the error  (b)(1)(i)
    - Written confirmation of an oral notice may be required within 10 business days     (b)(2)
    - Determine within 10 business days, or provisionally credit within 10 business days
      and finish within 45 days                                                           (c)(1), (c)(2)
    - 20 business days for a transfer within 30 days after the first deposit             (c)(3)(i)
    - 90 days instead of 45 for a transfer not initiated within a state, a point-of-sale
      debit card transaction, or a transfer within 30 days after the first deposit       (c)(3)(ii)
  Reg Z, unauthorized use (1026.12(b))
    - At most the lesser of $50 or the use before notice, only if the (b)(2) conditions hold;
      no reporting deadline                                                              (b)(1)(ii), (b)(2)
  Reg Z, billing errors (1026.13)
    - Notice received no later than 60 days after the first statement showing the error (b)(1)
    - Acknowledge within 30 days of receiving it                                         (c)(1)
    - Resolve within 2 complete billing cycles, never later than 90 days                 (c)(2)
      "Two complete billing cycles" are two actual cycles after receipt: a notice received
      during a cycle gets the rest of that cycle plus the next two                       comment 13(c)(2)-1

Day counting (one convention, used everywhere):
  - "within N days after D" ends on D + N calendar days (D itself is day 0);
  - "within N business days after D" ends on the Nth business day after D; D itself is never
    counted (comment 6(b)(1)-3), and the period ends at 11:59 p.m. that day.
Business days are Monday to Friday except the weekday holiday closures in
data/us_federal_holidays.json (Federal Reserve schedule, 2025-2027). That is an assumption: Reg E's
business day is a day the customer's own institution is open (1005.2(d)). Every result says so.

Arguments that don't fit the regulation, or a date the holiday table doesn't cover, return
{"error": {"argument": ..., "message": ...}} instead of a guess.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOLIDAYS_FILE = REPO / "data" / "us_federal_holidays.json"

TRANSFER_KINDS = ("standard", "pos_debit", "foreign")

BUSINESS_DAY_ASSUMPTION = (
    "Business days are Monday to Friday except Federal Reserve holiday closures (2025-2027). "
    "Reg E counts the days the customer's own institution is open (1005.2(d)), so for a bank that "
    "opens on Saturdays or holidays, business-day deadlines can fall earlier.")
DAY_COUNT = (
    "'Within N days' ends N calendar days after the start date; 'within N business days' ends on "
    "the Nth business day after it, at 11:59 p.m. The start date itself is never counted.")


class ToolInputError(ValueError):
    def __init__(self, argument: str, message: str):
        super().__init__(message)
        self.argument = argument


# ------------------------------------------------------------------ calendar

def load_holidays(path: Path = HOLIDAYS_FILE) -> tuple[set[date], date, date]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    closed = {date.fromisoformat(h["date"]) for h in data["closed"]}
    first, last = (date.fromisoformat(d) for d in data["covers"])
    return closed, first, last


CLOSED, COVERS_FROM, COVERS_TO = load_holidays()


def is_business_day(d: date, argument: str) -> bool:
    if not COVERS_FROM <= d <= COVERS_TO:
        raise ToolInputError(argument, f"business days can only be counted from {COVERS_FROM} to "
                                       f"{COVERS_TO} (the holiday table's range); {d} is outside it")
    return d.weekday() < 5 and d not in CLOSED


def add_business_days(start: date, n: int, argument: str) -> date:
    """The Nth business day after `start` (start itself is never counted)."""
    d, counted = start, 0
    while counted < n:
        d += timedelta(days=1)
        if is_business_day(d, argument):
            counted += 1
    return d


def add_days(start: date, n: int) -> date:
    return start + timedelta(days=n)


# ------------------------------------------------------------------ inputs

def parse_date(value, argument: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ToolInputError(argument, f"{argument} must be a date written YYYY-MM-DD, got {value!r}") from None


def reject(argument: str, value, why: str) -> None:
    if value not in (None, []):
        raise ToolInputError(argument, f"{argument} {why}")


def deadline(what: str, when: date, rule: str, party: str, met: bool | None = None) -> dict:
    out = {"what": what, "date": when.isoformat(), "rule": rule, "party": party}
    if party == "customer":
        out["met"] = met
    return out


# ------------------------------------------------------------------ the rules

def reg_e_unauthorized(notice: date, learned: date | None, statement: date | None) -> dict:
    tiers, deadlines, notes = [], [], []
    if learned is None and statement is None:
        raise ToolInputError("learned_of_loss", "give learned_of_loss (lost or stolen card or other access "
                                                "device) or statement_sent (transfer shown on a statement), or both")
    if learned is not None:
        end = add_business_days(learned, 2, "learned_of_loss")
        met = notice <= end
        deadlines.append(deadline("Notify within 2 business days after learning of the loss or theft, to keep "
                                  "liability at most $50", end, "1005.6(b)(1)", "customer", met))
        if met:
            tiers.append("max_50")
            notes.append("Liability is at most the lesser of $50 or the unauthorized transfers before notice "
                         "(1005.6(b)(1)).")
        else:
            tiers.append("max_500")
            notes.append("Liability is at most the lesser of $500 or the sum of up to $50 of transfers within the "
                         "2 business days plus transfers after them and before notice that the institution shows "
                         "timely notice would have prevented (1005.6(b)(2)).")
    else:
        notes.append("No lost or stolen access device given: the $50 and $500 tiers don't apply "
                     "(comment 6(b)(3)-2).")
    if statement is not None:
        end = add_days(statement, 60)
        met = notice <= end
        deadlines.append(deadline("Report unauthorized transfers shown on the statement within 60 days of its "
                                  "transmittal", end, "1005.6(b)(3)", "customer", met))
        if not met:
            tiers.append("later_transfers_unprotected")
            notes.append(f"The customer can also be liable for transfers after {end.isoformat()} and before notice "
                         "that the institution shows timely notice would have prevented (1005.6(b)(3)).")
        elif learned is None:
            tiers.append("none")
    else:
        notes.append("No statement date given: the 60-day statement rule (1005.6(b)(3)) was not checked.")
    if any(d["met"] is False for d in deadlines):
        notes.append("If the delay was due to extenuating circumstances, such as extended travel or a hospital "
                     "stay, the institution must extend these time limits to a reasonable period (1005.6(b)(4)). "
                     "This tool can't judge that.")
    notes.append("State law or the account agreement may set lower liability, which then applies (1005.6(b)(6)); "
                 "not checked here.")
    return {"liability_tiers": tiers, "deadlines": deadlines,
            "notice_timely": all(d["met"] for d in deadlines), "notes": notes}


def reg_e_error(notice: date, statement: date | None, first_deposit: date | None,
                transfer: date | None, kind: str | None) -> dict:
    if kind not in TRANSFER_KINDS:
        raise ToolInputError("transfer_kind", f"transfer_kind must be one of {', '.join(TRANSFER_KINDS)} "
                                              f"for a Reg E error, got {kind!r}")
    if first_deposit is not None and transfer is None:
        raise ToolInputError("transfer_date", "transfer_date is needed with first_deposit: the new-account periods "
                                              "depend on whether the transfer was within 30 days after it")
    if first_deposit is not None and transfer < first_deposit:
        raise ToolInputError("transfer_date", "transfer_date is before first_deposit")
    deadlines, notes, timely = [], [], None
    if statement is not None:
        end = add_days(statement, 60)
        timely = notice <= end
        deadlines.append(deadline("Notice of error received no later than 60 days after the statement first "
                                  "showing it was sent", end, "1005.11(b)(1)(i)", "customer", timely))
        if not timely:
            notes.append("The notice came more than 60 days after the statement, so the institution need not follow "
                         "the 1005.11 error procedures. If the error is an unauthorized transfer, it must still "
                         "apply 1005.6 before holding the customer liable (comment 11(b)(1)-7).")
            return {"liability_tiers": None, "deadlines": deadlines, "notice_timely": False, "notes": notes}
    else:
        notes.append("No statement date given: whether the notice was in time (60 days after the statement, "
                     "1005.11(b)(1)(i)) was not checked.")

    new_account = first_deposit is not None and transfer <= add_days(first_deposit, 30)
    short = 20 if new_account else 10
    long = 90 if (new_account or kind in ("pos_debit", "foreign")) else 45
    short_end = add_business_days(notice, short, "notice_date")
    deadlines.append(deadline("If the institution requires written confirmation of an oral notice, it is due "
                              "within 10 business days", add_business_days(notice, 10, "notice_date"),
                              "1005.11(b)(2)", "customer"))
    short_rule = "1005.11(c)(3)(i)" if new_account else "1005.11(c)(1)"
    deadlines.append(deadline(f"Determine whether an error occurred within {short} business days of receiving "
                              "the notice", short_end, short_rule, "institution"))
    deadlines.append(deadline(f"Or, to take longer: provisionally credit the account within {short} business days",
                              short_end, "1005.11(c)(2)(i)" + (", (c)(3)(i)" if new_account else ""), "institution"))
    long_rule = "1005.11(c)(3)(ii)" if long == 90 else "1005.11(c)(2)"
    deadlines.append(deadline(f"Finish the investigation within {long} days of receiving the notice (only with "
                              "provisional credit given)", add_days(notice, long), long_rule, "institution"))
    if new_account:
        notes.append("New account: the transfer was within 30 days after the first deposit, so 20 business days "
                     "and 90 days apply (1005.11(c)(3)(i), (c)(3)(ii)(C)).")
    elif first_deposit is None:
        notes.append("No first_deposit given: treated as not a new account.")
    if kind == "pos_debit":
        notes.append("Point-of-sale debit card transaction: 90 days instead of 45 (1005.11(c)(3)(ii)(B)).")
    if kind == "foreign":
        notes.append("Transfer not initiated within a state: 90 days instead of 45 (1005.11(c)(3)(ii)(A)).")
    notes.append("After finishing, the institution reports the results within 3 business days and corrects an "
                 "error within 1 business day of finding it (1005.11(c)(1), (c)(2)(iii)-(iv)); it tells the customer "
                 "about a provisional credit within 2 business days of making it (1005.11(c)(2)(ii)). These depend "
                 "on dates the tool isn't given.")
    return {"liability_tiers": None, "deadlines": deadlines, "notice_timely": timely, "notes": notes}


def reg_z_unauthorized() -> dict:
    return {
        "liability_tiers": ["max_50"],
        "deadlines": [],
        "notice_timely": None,
        "notes": [
            "Liability is at most the lesser of $50 or the use before notice to the card issuer (1026.12(b)(1)(ii)). "
            "Reg Z sets no deadline for reporting unauthorized use.",
            "The cardholder is liable at all only if (i) the card is an accepted credit card, (ii) the issuer gave "
            "adequate notice of the maximum liability and how to report loss or theft, and (iii) the issuer provides "
            "a means to identify the cardholder or authorized user (1026.12(b)(2)). This tool can't check these.",
            "This protection doesn't depend on the billing-error procedures or their time limits in 1026.13 "
            "(comment 12(b)(3)-3).",
            "State law or the cardholder agreement may set lower liability, which then applies (1026.12(b)(4)); "
            "not checked here.",
        ],
    }


def reg_z_error(notice: date, statement: date | None, closings: list[date] | None) -> dict:
    deadlines, notes, timely = [], [], None
    if statement is not None:
        end = add_days(statement, 60)
        timely = notice <= end
        deadlines.append(deadline("Written billing error notice received no later than 60 days after the first "
                                  "statement showing the error was sent", end, "1026.13(b)(1)", "customer", timely))
        if not timely:
            notes.append("The notice came more than 60 days after the statement, so it isn't a billing error notice "
                         "under 1026.13(b)(1) and the 1026.13 duties don't apply. Protection against unauthorized "
                         "use in 1026.12(b) doesn't depend on it (comment 12(b)(3)-3).")
            return {"liability_tiers": None, "deadlines": deadlines, "notice_timely": False, "notes": notes}
    else:
        notes.append("No statement date given: whether the notice was in time (60 days after the first statement, "
                     "1026.13(b)(1)) was not checked.")

    deadlines.append(deadline("Mail or deliver a written acknowledgment within 30 days of receiving the notice "
                              "(unless the error is resolved within that time)", add_days(notice, 30),
                              "1026.13(c)(1)", "institution"))
    cap = add_days(notice, 90)
    complete = [c for prev, c in zip(closings or [], (closings or [])[1:]) if prev >= notice]
    if len(complete) >= 2:
        two_cycles = complete[1]
        resolve = min(two_cycles, cap)
        notes.append(f"Two complete billing cycles after receipt end on {two_cycles.isoformat()}; 90 days end on "
                     f"{cap.isoformat()}; the earlier date applies. The cycle in which the notice arrived doesn't "
                     "count (comment 13(c)(2)-1).")
    else:
        resolve = cap
        notes.append("Not enough billing cycle closing dates to find the end of two complete cycles after receipt "
                     "(give the first closing date on or after the notice date and the two after it). Showing the 90-day "
                     "outer limit only; the two-cycle date may be earlier.")
    deadlines.append(deadline("Resolve within 2 complete billing cycles after receiving the notice, never later "
                              "than 90 days", resolve, "1026.13(c)(2)", "institution"))
    notes.append("A billing error notice must be written and sent to the address the creditor discloses for it "
                 "(1026.13(b)); a phone call alone isn't one.")
    return {"liability_tiers": None, "deadlines": deadlines, "notice_timely": timely, "notes": notes}


# ------------------------------------------------------------------ entry point

def dispute_timeline(regulation: str, event: str, notice_date, learned_of_loss=None, statement_sent=None,
                     first_deposit=None, transfer_date=None, transfer_kind=None, cycle_closing_dates=None) -> dict:
    """Deadlines and liability caps for a Reg E or Reg Z dispute. Dates are `datetime.date` or 'YYYY-MM-DD'.
    Returns the result, or {"error": {"argument", "message"}} for input that doesn't fit."""
    try:
        return _timeline(regulation, event, notice_date, learned_of_loss, statement_sent, first_deposit,
                         transfer_date, transfer_kind, cycle_closing_dates)
    except ToolInputError as e:
        return {"error": {"argument": e.argument, "message": str(e)}}


def _timeline(regulation, event, notice_date, learned_of_loss, statement_sent, first_deposit,
              transfer_date, transfer_kind, cycle_closing_dates) -> dict:
    if regulation not in ("E", "Z"):
        raise ToolInputError("regulation", f"regulation must be 'E' or 'Z', got {regulation!r}")
    if event not in ("unauthorized", "error"):
        raise ToolInputError("event", f"event must be 'unauthorized' or 'error', got {event!r}")
    notice = parse_date(notice_date, "notice_date")
    if notice is None:
        raise ToolInputError("notice_date", "notice_date is required")
    learned = parse_date(learned_of_loss, "learned_of_loss")
    statement = parse_date(statement_sent, "statement_sent")
    deposit = parse_date(first_deposit, "first_deposit")
    transfer = parse_date(transfer_date, "transfer_date")
    if cycle_closing_dates is not None and not isinstance(cycle_closing_dates, (list, tuple)):
        raise ToolInputError("cycle_closing_dates", "cycle_closing_dates must be a list of dates")
    closings = [parse_date(c, "cycle_closing_dates") for c in cycle_closing_dates or []]
    if any(b <= a for a, b in zip(closings, closings[1:])):
        raise ToolInputError("cycle_closing_dates", "cycle_closing_dates must be in order with no repeats")

    if regulation == "Z":
        for name, value in (("learned_of_loss", learned), ("first_deposit", deposit),
                            ("transfer_date", transfer), ("transfer_kind", transfer_kind)):
            reject(name, value, "applies to Regulation E only")
    else:
        reject("cycle_closing_dates", closings, "applies to Regulation Z billing errors only")
    if event == "unauthorized":
        for name, value in (("first_deposit", deposit), ("transfer_date", transfer),
                            ("transfer_kind", transfer_kind), ("cycle_closing_dates", closings)):
            reject(name, value, "applies to event 'error' only")
    else:
        reject("learned_of_loss", learned, "applies to event 'unauthorized' only")
    if regulation == "Z" and event == "unauthorized":
        reject("statement_sent", statement, "doesn't apply: Reg Z sets no reporting deadline for unauthorized use")

    if regulation == "E" and event == "unauthorized":
        body = reg_e_unauthorized(notice, learned, statement)
    elif regulation == "E":
        body = reg_e_error(notice, statement, deposit, transfer, transfer_kind)
    elif event == "unauthorized":
        body = reg_z_unauthorized()
    else:
        body = reg_z_error(notice, statement, closings)
    return {"regulation": regulation, "event": event, "notice_date": notice.isoformat(), **body,
            "assumptions": [BUSINESS_DAY_ASSUMPTION, DAY_COUNT]}


# ------------------------------------------------------------------ the tool as the model sees it

TOOL_SPEC = {
    "toolSpec": {
        "name": "dispute_timeline",
        "description": (
            "Works out dispute deadlines and liability caps from dates, for Regulation E (debit cards, ATM, ACH "
            "and other electronic transfers from a bank account) and Regulation Z (credit cards). Call it for any "
            "deadline or liability amount that depends on dates; never count days yourself. "
            "event 'unauthorized': Reg E liability tiers (1005.6(b)) or the Reg Z $50 cap (1026.12(b)). "
            "event 'error': Reg E error resolution (1005.11) or Reg Z billing errors (1026.13). "
            "Business days assume Monday to Friday without Federal Reserve holidays, dates 2025 to 2027. "
            "Not for prepaid accounts without periodic statements (1005.18(e)). "
            "Returns an error naming the argument when the inputs don't fit."),
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "regulation": {"type": "string", "enum": ["E", "Z"]},
                "event": {"type": "string", "enum": ["unauthorized", "error"],
                          "description": "Reg Z 'error' means a billing error"},
                "notice_date": {"type": "string", "format": "date",
                                "description": "When the customer notified the institution (YYYY-MM-DD)"},
                "learned_of_loss": {"type": "string", "format": "date",
                                    "description": "Reg E unauthorized only: when the customer learned the card "
                                                   "or other access device was lost or stolen"},
                "statement_sent": {"type": "string", "format": "date",
                                   "description": "When the first statement showing the problem was sent "
                                                  "(not for Reg Z unauthorized)"},
                "first_deposit": {"type": "string", "format": "date",
                                  "description": "Reg E error only, for a new account: date of the first deposit"},
                "transfer_date": {"type": "string", "format": "date",
                                  "description": "Reg E error only: date of the disputed transfer; needed with "
                                                 "first_deposit"},
                "transfer_kind": {"type": "string", "enum": list(TRANSFER_KINDS),
                                  "description": "Reg E error only, required: 'pos_debit' for any debit card "
                                                 "purchase (including mail and phone orders), 'foreign' for a "
                                                 "transfer not initiated within a state, otherwise 'standard' "
                                                 "(ATM transactions are 'standard')"},
                "cycle_closing_dates": {"type": "array", "items": {"type": "string", "format": "date"},
                                        "description": "Reg Z billing error only: consecutive billing cycle "
                                                       "closing dates: the first one on or after the notice date "
                                                       "and the ones after it"},
            },
            "required": ["regulation", "event", "notice_date"],
        }},
    }
}
