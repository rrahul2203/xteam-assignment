# Support Ticket Router — baseline review

Numbers below come from `notebook/baseline_classifier_training.ipynb`.

## Verdict

No sign-off yet. The model looks fine; the evaluation can't back the claim.

The report notices `fraud-report` has only 50 examples, then waves it off because accuracy
is 98.75%. Accuracy can't answer a question about the rarest class — that's the part to fix.

## Evaluation changes

**Fit the vectoriser after the split.** `fit_transform(texts)` runs before
`train_test_split`, so IDF weights see the test rows. Worth fixing, but it isn't inflating
anything here — corrected, the score goes up.

**Drop the single 80/20 split.** 80 test messages means one misroute swings the headline
1.25 points. "98.75%" and "one message wrong" are the same sentence.

**Split by template.** Strip numbers, asset names and greetings and 400 messages collapse
to 256 intents; "how do i enable price alerts for ___" appears 9 times. A random split puts
siblings on both sides. 62 of 80 test rows sit within 0.80 cosine of a training row.
`StratifiedGroupKFold` fixes it.

**Report per-class recall.** `general` is 40% of the data and drowns everything else.

**Test on messages nobody templated.** Missing, and the only real fix.

## The honest number

| Protocol | Accuracy |
|---|---|
| Reported (leaky, one split) | 0.9875 |
| Leakage fixed | 1.0000 |
| Random 5-fold | 1.0000 |
| Grouped 5-fold, no shared template | 1.0000 |

100%, which is worse news than 98.75%. I built the grouped split expecting it to break the
model and it didn't. Nothing in `train.csv` produces an error, so the set can't rank two
models or catch a regression.

A 20-word vocabulary still scores 83.3%; 30 words, 84.8%. Mostly keyword lookup.

So: accuracy is unmeasured. 98.75% isn't wrong, it's unfalsifiable. First deliverable is a
few hundred real messages labelled by someone who didn't write the model.

Hand-written paraphrases scored 8/9 — encouraging, and only 9 probes, so it stays an
anecdote.

## Production metric

**`fraud-report` recall ≥ 0.98 on non-templated traffic, weekly, with a CI.**

- Recall gates the release. Accuracy doesn't.
- Precision is a budget, floor ~0.60, spend the rest on recall. A 5× fraud class weight is
  the dial; on this data it changes nothing, so it needs the harder set.
- Below ~0.60 confidence, send to a human. Fraud probes averaged 0.60 against 0.83
  elsewhere — the model is least sure exactly where it's most expensive.
- Flag fraud vocabulary even when the route is something else. The one probe it missed,
  *"I can't log in AND I see a transfer I never made"*, went to `account-access` at 0.46.
  Real queues are full of mixed intent and one label can't say "both".
- Watch how far incoming text drifts from training. Labels arrive late; drift arrives first.

Secondary: macro-F1, per-class recall, abstention rate, p95 latency.

## Before shipping

1. Vectoriser into a `Pipeline`.
2. Grouped CV and per-class recall as the standard report.
3. Confidence threshold plus fraud-keyword net.
4. Shadow mode behind the queue, recall on a dashboard.

Baseline's probably close to good enough. The evaluation isn't.
