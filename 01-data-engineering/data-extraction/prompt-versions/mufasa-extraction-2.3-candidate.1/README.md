# mufasa-extraction-2.3-candidate.1

The Cavoti/Claude Opus production-candidate prompts after the exact-grounding
contract repair. These files are rendered exactly as the model receives them.

Relative to 2.2-candidate.2, this version makes literal source copying,
multi-atom source grouping, alias uniqueness, context ownership, observation
grounding, and reranker grounding explicit. Repair calls also receive the
rejected JSON and every validator error; that repair protocol lives in the
notebook's user-prompt builder rather than these system-prompt files.

