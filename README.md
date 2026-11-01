# RegZE Assistant

An internal assistant for staff at a US card issuer and bank. It answers questions about two US consumer-finance regulations, cites the exact paragraph it relies on, works out dispute deadlines and liability limits with a deterministic tool, and declines when the regulations don't answer the question.

- **Regulation Z** (Truth in Lending, 12 CFR Part 1026): credit cards and other open-end credit.
- **Regulation E** (Electronic Fund Transfers, 12 CFR Part 1005): debit cards, ATM and other electronic transfers from bank accounts.

The regulation text comes from the official eCFR API as XML for one "as of" date, recorded in `corpus/manifest.json`. US federal regulations are works of the US government and are not under copyright.

Status: work in progress. Nothing has been evaluated yet, and no results are claimed.

Not legal or compliance advice. Card network rules, state law and an issuer's own account terms are out of scope.

## History

This repository began as an Indian law Q&A project (commits before the switch to Regulations Z and E). That corpus and pipeline were retired; the embedding and search code carried over.
