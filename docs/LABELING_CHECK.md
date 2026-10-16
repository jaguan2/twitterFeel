# Actual-model labeling check — October 16, 2026

The diagnostic runs 12 synthetic posts through the labeling CLI and the native
desktop bridge's actual inference action. Desktop demonstrations and dataset jobs
share `AnalysisEngine`; temporary app state keeps research annotations untouched.
The separate native smoke check tests the window and its interactive controls.

This is an execution and consistency diagnostic, **not an accuracy benchmark**.
The scenarios and comments below are provisional project judgments. There are
no independent human gold labels, adjudication, or accuracy claims.

The current run is `runs/labeling_diagnostic_desktop_v1/`. Earlier runs under
`runs/labeling_diagnostic_v1/` through `v3/` checked the retired browser UI.
The **Experiments → Model execution checks** section displays these artifacts
in the app. Matching execution does not resolve the model weaknesses below.

## Observed predictions

Shared operating settings: threshold **0.5**, maximum **256 tokens**, CLI batch
size **1**, CPU inference. All 11 nonblank inputs agreed within the diagnostic's
**0.00001** score tolerance; the maximum absolute difference was approximately
**0.0000000103**. Selected labels and result statuses matched. Blank input was
marked invalid by the CLI and rejected by the app.

| Scenario | Synthetic input | Selected model labels |
| --- | --- | --- |
| Mixed anticipation | I'm excited about moving, but terrified of leaving everyone behind. | fear |
| Mixed farewell | I'm happy for your promotion, but sad that you are leaving us. | sadness |
| Factual statement | The meeting starts at three in room 204. | neutral |
| Negation | I'm not angry; I'm disappointed. | disappointment |
| Sarcasm | Great, another cancelled train. Just what I needed. | admiration |
| Quotation | She said "I am terrified." The interview transcript is attached. | neutral |
| Ambiguous expression | Well, that was something. | neutral |
| Blank | Empty string | none; invalid |
| Spanish | Estoy feliz de verte, pero tengo miedo del viaje. | none; uncertain |
| Explicit joy | I feel so happy and joyful today! | joy |
| Explicit fear | I am terrified and afraid for my safety. | fear |
| Truncation | "I am excited about this." repeated 90 times | excitement; CLI review needed |

The CLI summary was six labeled, three neutral, one uncertain, one invalid,
and one needing review. These counts describe this constructed set only.

## What needs attention

1. **Supporting multiple labels does not guarantee detecting mixed emotions.**
   Neither mixed example retained both explicitly expressed emotions at 0.5.
   Compare threshold settings on reviewed development data before changing the
   default; lowering thresholds can also introduce false labels.
2. **Sarcasm remains a concrete weakness.** The cancelled-train example received
   admiration. Include sarcasm in the separate challenge set and consider
   context-aware review. A word such as "Great" alone is insufficient evidence.
3. **Ambiguity can be mislabeled neutral.** The context-dependent example received
   neutral, despite the annotation rubric treating it as uncertain. Independent
   model scores and threshold abstention do not solve missing-context detection.
4. **Unsupported language needs an explicit policy.** Spanish happened to abstain
   here. That is not evidence of Spanish competence or reliable language
   detection; language screening or human review remains to be added.
5. **Truncation requires review.** Both paths mark the long post `review_needed`.
   The shared engine and native smoke checks cover this behavior.

Long-input token counting also printed a Transformers length warning before
actual inference. The model received the correctly truncated 256-token input
and completed successfully; the warning came from the preliminary full-length
token count. This can be clarified or suppressed specifically for length
inspection without hiding inference errors.

## Reproduce and inspect artifacts

```powershell
.venv-workbench/Scripts/python.exe src/check_labeling.py --out runs/labeling_diagnostic_desktop_v2
```

Use a fresh output name. The diagnostic script runs the actual labeling CLI,
then actual app inference, and saves `diagnostics.json`, exact synthetic inputs,
CLI predictions, scores, runtime configuration, and source/model fingerprints.
The command exits with a failure code when execution/score/label/status parity
checks fail. It does not score the quality of the model's emotion judgments.
Full scores, source/engine hashes, model fingerprints, and CLI configuration
are retained in each run's JSON artifacts. The table above reports selected
labels only. This diagnostic uses the RoBERTa baseline; the separate
[refinement report](THRESHOLD_REFINEMENT.md) compares it with ModernBERT.

The next quality gate is the blind human-review benchmark described in
[EMOTION_LABELING.md](EMOTION_LABELING.md), followed by threshold/model comparison
on development posts and a final held-out evaluation.
