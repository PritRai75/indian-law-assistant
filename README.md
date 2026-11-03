# RegZE Assistant

An internal assistant for staff at a US card issuer and bank. It answers questions about two US consumer-finance regulations, cites the exact paragraph it relies on, works out dispute deadlines and liability limits with a deterministic tool, and declines when the regulations don't answer the question.

- **Regulation Z** (Truth in Lending, 12 CFR Part 1026): credit cards and other open-end credit.
- **Regulation E** (Electronic Fund Transfers, 12 CFR Part 1005): debit cards, ATM and other electronic transfers from bank accounts.

The regulation text comes from the official eCFR API as XML for one "as of" date, recorded in `corpus/manifest.json`. US federal regulations are works of the US government and are not under copyright.

Status: work in progress. Nothing has been evaluated yet, and no results are claimed.

Not legal or compliance advice. Card network rules, state law and an issuer's own account terms are out of scope.

## Corpus pipeline (so far)

| Step | Command | Output |
| --- | --- | --- |
| Download | `python -m ingest.fetch` | `corpus/raw/part-1005.xml`, `part-1026.xml`, `corpus/manifest.json` (as-of date, hashes) |
| Parse | `python -m ingest.parse` | `corpus/txt/provisions.jsonl`: one record per rule paragraph or numbered official interpretation, with its citation ID |
| Chunk | `python -m ingest.chunk` | `corpus/txt/chunks.jsonl`: the 1,115 search units (one section, paragraph, or set of comments on one paragraph, up to 4,000 characters), each with a header naming its source and status |
| Compare versions | `python -m ingest.compare` | Which provisions changed at each eCFR amendment date (used for the status register, and when the corpus is refreshed) |

**Provision IDs.** Rule paragraphs use their CFR citation (`1026.52(b)(1)(ii)(A)`); official interpretations use the part plus the comment number (`1026.52(b)(1)(ii)-1` is Reg Z comment 52(b)(1)(ii)-1).

**Source defects handled.** The eCFR XML omits one commentary heading (Reg E 17(b)(3)) and repeats a comment number (Reg Z 11(c)); the parser corrects both to match the CFPB's published commentary and logs them. It stops with an error on any label or comment number it can't place, rather than guessing.

**Text the eCFR prints that is not law in force.** The eCFR shows amendments as of their effective date even after a court vacates them or Congress overturns them, and the XML doesn't mark them. `corpus/status.json` is a hand-checked register of such provisions (status, reason, source), and every provision carries its `status`, so the assistant can decline with an explanation instead of stating a vacated rule as law. A search chunk never mixes statuses, so a vacated paragraph can't hide inside an in-force chunk. One paragraph that the eCFR no longer prints but that is still law (12 CFR 1026.2(a)(15)(ii)(C), removed by the overturned overdraft rule) is added as its own chunk with a note saying why.

## History

This repository began as an Indian law Q&A project (commits before the switch to Regulations Z and E). That corpus and pipeline were retired; the embedding and search code carried over.
