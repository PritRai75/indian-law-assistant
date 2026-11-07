"""Run settings in one place (spec: model IDs in one config file). Every eval run logs `config_hash`,
which covers these settings, the prompt version and the corpus "as of" date, so two runs with the same
hash were made the same way.

Model IDs are the global inference profiles tested on Prit's account in session 2.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, replace

from app.prompts import PROMPT_VERSION, PROMPTS, TOOL_OFF_ONLY

# The citation guard's rule (app.answer.guard). Bump it with any change to how the guard reads or
# removes citations, since that changes answers. "c1": one ID per string (sessions 12-14);
# "c2": every citation read with app.citations (packed IDs, ranges, comment sub-points; session 15);
# "c3": also comment sub-points written in brackets, "1026.55(b)-2(ii)" (session 15).
GUARD_VERSION = "c3"


@dataclass(frozen=True)
class Config:
    generator_model: str = "global.anthropic.claude-sonnet-4-6"
    judge_model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    temperature: float = 0.0
    max_tokens: int = 1024            # a 2-6 sentence answer as JSON needs a few hundred
    k: int = 5                        # passages given to the model
    search_mode: str = "hybrid"       # vector, bm25 or hybrid (experiment 2)
    tool_on: bool = True              # experiment 3 turns it off
    max_tool_turns: int = 2           # model turns that may call the tool; a third tool request ends the loop
    output_mode: str = "json"         # "json": the prompt asks for JSON and Pydantic checks it, one repair try;
                                      # "structured": Bedrock structured outputs enforce the schema as well
    read_timeout: int = 120           # seconds; structured outputs can take ~90 s to compile a new schema
    max_question_chars: int = 2000
    prompt: str = PROMPT_VERSION      # which app.prompts.PROMPTS text the answer model gets (experiment 4)
    interpretations: bool = True      # search the official interpretations too; False: rule text only (experiment 1)

    def __post_init__(self):
        if self.search_mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"search_mode must be vector, bm25 or hybrid, not {self.search_mode!r}")
        if self.prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {', '.join(PROMPTS)}, not {self.prompt!r}")
        if self.prompt in TOOL_OFF_ONLY and self.tool_on:
            raise ValueError(f"prompt {self.prompt} is written for runs without the tool: use it with the tool off")
        if self.output_mode not in ("json", "structured"):
            raise ValueError(f"output_mode must be json or structured, not {self.output_mode!r}")

    def with_(self, **changes):
        return replace(self, **changes)


DEFAULT = Config()

# Settings added after runs were first logged (session 22): left out of the hash at their default
# value, so a config that doesn't use them hashes as it did before they existed.
ADDED_LATER = {"interpretations": True}


def config_hash(config, as_of):
    """First 12 hex digits of a hash over the settings, the prompt version, GUARD_VERSION and the corpus
    "as of" date. The `prompt` setting enters as "prompt_version", as it did before it was a setting, and
    settings in ADDED_LATER count only when they differ from their default, so a default config hashes
    as it did before session 22 (earlier runs stay comparable by hash)."""
    settings = {k: v for k, v in asdict(config).items()
                if k != "prompt" and not (k in ADDED_LATER and v == ADDED_LATER[k])}
    blob = json.dumps({**settings, "prompt_version": config.prompt, "guard_version": GUARD_VERSION,
                       "as_of": as_of}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
