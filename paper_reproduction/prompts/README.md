# V14 agent system prompt: the four versions behind Table 1

The second (agent) stage of BioHarness runs a REPL agent whose system prompt is the
string `V14_SYSTEM_PROMPT` in `src/rlm/pipeline.py`. That string changed three times
between the April runs and the last LitQA2 re-run, so the Ours row of Table 1 was
produced by four different prompt texts. Each file here holds one version exactly, as
the Python string value, with nothing added.

| File | Version | In force | Length (chars) |
|---|---|---|---|
| `V_apr.txt` | April | until 2026-04-27 06:41 UTC | 2,658 |
| `V_pre.txt` | pre-audit: `V_apr` plus the `atlas` context block | 2026-04-27 to 2026-09-09 03:52 UTC | 4,304 |
| `V_audit.txt` | the 2026-09-09 prompt audit | 2026-09-09 03:52 UTC to 2026-09-10 02:27 UTC | 4,086 |
| `V_cur.txt` | `V_audit` plus two anti-abstention rules | from 2026-09-10 02:27 UTC | 4,662 |

Checks: `V_cur` − `V_pre` = 358 characters and `V_cur` − `V_audit` = 576 characters.
`V_cur.txt` is byte-identical to `V14_SYSTEM_PROMPT` in the released
`src/rlm/pipeline.py`. `V_pre.txt` and `V_audit.txt` are byte-identical to the
`V14_SYSTEM_PROMPT` strings in dated backups of `pipeline.py` taken on 2026-09-09 (before
the audit edit) and 2026-09-10 (before the anti-abstention edit).

What changed:

- `V_apr` → `V_pre`: adds the description of the `atlas` variable (scRNA/HPA atlas
  context, vocabulary alignment) used by the atlas-routed configuration.
- `V_pre` → `V_audit`: removes the "CRITICAL RULES" block, softens "NEVER answer 'not
  found' without trying at least one tool" to "Try at least one tool before answering
  'not found'", and rewrites the output-format section around `FINAL("...")`.
- `V_audit` → `V_cur`: adds "the evidence you were given came from one retrieval pass
  ... a reason to search, not a reason to stop", "search the full-text tools before
  `search_papers`", and "Never answer 'insufficient information' / 'cannot be
  determined' without having run at least one retrieval call in this session that
  returned results".

## Which Table 1 cells each version produced

| Version | Table 1 cells (Ours row) | Run script |
|---|---|---|
| `V_apr` | the eight original columns: PubMedQA 76.4, BioASQ 54.1, MedMCQA 74.9, MedQA-US 85.9, MedQA-TW 89.3, MedQA-CN 89.7, and the unspliced GeneTuring / SciHorizon values (28.8 / 56.0) | `runs/ours_april_eight_columns.sh` |
| `V_pre` | GeneTuring 54.6 (June 15 genomics re-run); the DISCO-routed items merged into BioASQ (April 28; the routed SciHorizon items were later replaced by the case study); the LitQA2 base run whose 57 non-re-run items enter 61.8, and footnote b's 57.8 three-run mean | `runs/ours_geneturing_genomics.sh`, `runs/ours_disco_merge.sh`, `runs/ours_litqa2_agentft.sh` |
| `V_audit` | MedXpertQA 37.1 (2026-09-09 14:54 to 17:52 local time) | `runs/ours_medxpertqa.sh` |
| `V_cur` | LitQA2 61.8: the 142 re-run items (2026-09-11) | `runs/ours_litqa2_rerun142.sh` |

Notes:

- None of the 57 base LitQA2 items was escalated, so the agent prompt never ran on them;
  their answers do not depend on the prompt version.
- The SciHorizon 60.3 expression splice comes from `scripts/case_study_repair_context.py`,
  which has its own expression prompt and does not use the V14 prompt.
- `V_apr` for 2026-04-13 to 2026-04-21 is reconstructed from the compiled April 13
  `pipeline.py` and the edit history; no edit to the prompt is recorded in that window.
- Only `V14_SYSTEM_PROMPT` is versioned here. The tool documentation appended to it
  (`{tool_section}`) also changed on 2026-09-10 (corrected tool signatures); the released
  code carries the corrected tool documentation for every version.

## Using a version

The released code carries `V_cur`. Any other version is injected without editing code:

```bash
export XC_V14_PROMPT_FILE=prompts/V_apr.txt   # or V_pre.txt / V_audit.txt
```

`src/rlm/pipeline.py` then reads the file in place of the built-in string. The run
scripts in `runs/` set this variable for each cell.

The override was added to the research code on 2026-09-16, after every Table 1 run; none of the runs behind Table 1 had it, and with the
variable unset the code uses the built-in `V_cur` string exactly as before.
