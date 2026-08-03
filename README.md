# ML practical

Two parts, side by side. [Part A](#part-a--support-ticket-router-baseline-review) reviews the
handed-over route classifier. [Part B](#part-b--answering-from-the-knowledge-base) is the
date-aware question answering service.

```
src/router/    Part A — data · crossval · model · evaluate · predict · tune
src/qa/        Part B — kb · retriever · answerer · eval_answers · answer_cli
tests/         Part B — test_kb · test_answerer
eval/gold.csv  Part B — hand-labelled retrieval labels
```

# Part A — Support Ticket Router baseline review

Numbers below come from `notebook/baseline_classifier_training.ipynb`.

## Running it

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# Route classification, per the starter pack's output contract.
./venv/bin/python -m src.router.predict --input messages.csv --output predictions.csv
./venv/bin/python -m src.router.predict --text "I see a transfer I never made"   # one message

./venv/bin/python -m src.router.evaluate   # the honest scores below
./venv/bin/python -m src.router.tune       # re-derive the tuned settings
```

`predictions.csv` has `text` and `prediction`; `--confidence` adds the top-class probability,
which is what a review queue would threshold on.

`crossval` holds the grouped-CV protocol that `evaluate`, `model` and `tune` all score
through, so the search and the reported number cannot drift apart.

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

## What the fix changes

Working version in `src/router/`, walked through in `notebook/baseline_fix_walkthrough.ipynb`.
Each number below is measured under template-grouped CV.

1. **Vectorisers moved inside a `Pipeline`.** The baseline calls `fit_transform` on all
   400 rows before splitting. Now no split can fit a vectoriser on rows it will score.
2. **Template-grouped CV replaces the 80/20 split.** `template_key` collapses 400 rows to
   70 intents. Same baseline recipe scores 0.9783 macro-F1 on a random split and 0.7863
   grouped — the split choice, not the model, produced the 98.75%.
3. **Character n-grams added alongside word features.** 0.7925 → 0.8494 macro-F1, the
   largest single gain. Unseen phrasings match on substrings, and typos stop being unknown
   tokens.
4. **Class weights skewed toward fraud, not just balanced.** Fraud recall 0.600 unweighted,
   0.776 at inverse-frequency, 0.904 skewed past it. Macro-F1 stays ~0.85 across all three,
   so the baseline's headline metric can't see this choice at all.
5. **The skew is derived, not chosen.** `solve_skew` takes the fraud precision floor the
   queue can absorb (0.60) and returns the largest skew clearing it. It solves to 2.5 at
   char=(2,4) and 1.75 at (2,5) — a hardcoded weight would have gone stale silently.
6. **Hyperparameters come from a grouped grid search**, averaged over 5 seeds. One
   deliberate override: the search ranks `C=30` first (0.8578 vs 0.8537) but it sheds fraud
   recall (0.876 vs 0.904), and recall is the guardrail.
7. **Weights are derived per fold** from training rows only. Computing them once from all
   400 labels leaks the class distribution into every fold.
8. **Fraud recall is reported as its own line**, next to both split protocols, so the
   leakage gap stays visible on every run instead of being something you have to go looking
   for.

Result: macro-F1 0.84, `fraud-report` recall 0.90 at 0.61 precision. Below the 0.98 recall
target above — that gap is real and needs the non-templated set.

Baseline's probably close to good enough. The evaluation isn't.

---

# Part B — Answering from the knowledge base

No LLM, no API key, no network. TF-IDF retrieval over 31 documents, and answers are sentences
copied verbatim from the document cited. That is the grounding argument: the answer text is a
substring of the source, so it cannot drift from it, and `doc_ids` is checkable by opening the
file. `test_answer_text_is_copied_from_the_cited_document` asserts exactly this.

## Running it

```bash
./venv/bin/python -m src.qa.answer_cli --input starter/questions.csv --output answers.csv
./venv/bin/python -m src.qa.answer_cli --question "How long do I have to raise a dispute?" \
                                       --as-of 2026-03-01
./venv/bin/python -m src.qa.eval_answers   # the numbers below, plus ablations
./venv/bin/python -m pytest tests/ -q      # 52 tests
```

```python
from src.qa.answerer import AnswerService
service = AnswerService()                       # loads the KB once
result = service.answer("How long do I have to raise a dispute?", "2026-03-01")
result.text, result.doc_ids, result.status      # -> "...within 60 days...", ["kb-031"], "answered"
```

## Correct as at a date

`status` is never consulted when deciding what is in force. `status` describes *today*, and a
question can ask about any date — filtering on `status: superseded` is precisely how a system
answers a March 2026 question with a July 2026 policy. Only `effective_date` and `valid_until`
decide, both ends inclusive.

**Filter first, rank second.** Ranking the whole KB and then dropping out-of-window hits lets a
superseded document win and be discarded, leaving a worse in-force document unranked. The
candidate pool for a date is built before anything is scored.

Verified by walking every day of all 8 supersession lineages: no gaps, no overlaps, exactly one
version in force on every date. That is a test, not a claim
(`test_one_version_of_a_lineage_is_in_force_at_a_time`).

The dispute window is the clean demonstration — same question, two dates:

| `as_of` | Cited | Answer |
|---|---|---|
| 2026-06-30 | kb-031 | within **60** days |
| 2026-07-01 | kb-032 | within **30** days |

## Three outcomes, not two

Beyond answered and declined there is **lapsed**: the only relevant document expired before
`as_of` and nothing replaced it. "No — that promotion ran until 2026-03-31 and is no longer in
force" is a real answer carrying real information, and abstaining there throws away something
the system knows. Superseded documents are excluded from this path, since their replacement is
in force and will be retrieved normally.

## Not everything is answerable

Nine of 38 questions are declined, with empty `doc_ids` and plain words saying a human agent is
needed. Silence is the right default when the cost of a confidently wrong policy answer is a
customer acting on it.

The measured finding is that **the obvious signals do not work**. Cosine cannot separate
answerable from unanswerable: being length-normalised, a question sharing two words with a
short document scores like one sharing eight with a long one, and the distributions overlap
almost entirely (unanswerable 0.092–0.263 against answerable 0.101–0.477). The share of
question words missing from the KB fails too — it is reported as a diagnostic and thresholded
on nothing.

What separates them is *which* words are missing. Every unanswerable question contains a domain
noun absent from the whole corpus — margin, custodian, chargeback, electricity. Answerable ones
are missing only filler and dates. So abstention is a conjunction: a **long absent term**
(≥6 chars, absent from the KB) **and** low **salient coverage** (IDF-weighted share of the
question's vocabulary that the top document contains, <0.35). Requiring both keeps "how many
days did a customer have in March 2026" answerable while declining "what interest rate on a
crypto backed margin loan". It moved abstention recall 0.375 → 0.875.

A lapsed hit faces the same 0.35 bar but without the absent-term escape, and otherwise yields
to the best in-force document — announcing an unrelated expiry is a confidently wrong answer,
not a helpful one.

## How it was measured, and what it scores

`eval/gold.csv`: 38 hand-labelled rows carrying the expected `doc_ids`, a stratum, and for most
rows a regex the answer text must contain, so a right document quoting the wrong sentence still
fails.

| Metric | Score |
|---|---|
| Document accuracy (answerable, right doc cited) | **0.867** (26/30) |
| Wrong-answer rate (answered confidently, wrong doc) | **0.053** |
| Answer recall (answerable questions answered) | 0.933 |
| Abstention recall (unanswerable declined) | 0.875 |
| Fact accuracy (quoted text contains the expected value) | 0.692 |

Per stratum, with Wilson intervals — the normal approximation is useless at n=4:

| Stratum | Score | 95% CI |
|---|---|---|
| current | 19/22 = 0.864 | [0.67, 0.95] |
| historical | 3/4 = 0.750 | [0.30, 0.95] |
| lapsed | 4/4 = 1.000 | [0.51, 1.00] |
| unanswerable | 7/8 = 0.875 | [0.53, 0.98] |

One aggregate number would hide all four, and the strata fail for different reasons.

**Ablations, because they are what make the headline mean anything.** If switching the date
filter off does not move the score, the score is not measuring the thing this exercise is about:

| Configuration | Doc accuracy | Wrong-answer |
|---|---|---|
| Shipped | 0.867 | 0.053 |
| No date filter (rank whole KB) | 0.533 | 0.316 |
| No abstention (always answer top hit) | 0.867 | 0.105 |
| Coverage only, no absent-term conjunction | 0.767 | 0.026 |

Dropping the date filter costs a third of document accuracy and sextuples the wrong-answer
rate. The coverage-only row is the honest trade: it declines more, so it is wrong less often,
and it gets fewer answerable questions right.

`eval_answers.py` also prints a coverage threshold sweep, so 0.35 is visible as a point on a
curve rather than asserted.

## The four failures

Named rather than tuned away — a stopping rule matters more here than 30/30 on n=38.

- **q01** — "how do I withdraw my balance" cites the account-closure doc, not kb-013. The doc
  says "transfers out"; the question says "withdraw". A vocabulary gap, and the honest fix is
  synonyms or embeddings, not another threshold.
- **q12** — abstains on a March 2026 dispute question at coverage 0.32 against the 0.35 bar.
  The date logic is right; the confidence gate is one notch too tight.
- **q19** — cites kb-072 over kb-093, two documents that genuinely overlap in wording.
- **q38** — answers an unanswerable question whose vocabulary happens to be all KB words.

Two things were tested and **rejected**: a word/char feature weight sweep (flat across every
weighting) and sentence-level scoring instead of document-level (worse, 25/30 against 26/30).
Recorded rather than shipped.

## Limitations

The same person wrote the system and the labels, and n=38 — every interval above is wide enough
to say so. This measures whether the date logic works, which it does. It cannot support a claim
about generalisation to unseen questions. TF-IDF also matches only on shared words, so q01 is
the whole class of failure it will keep having; sentence embeddings are the next step, and they
would slot behind the same `Retriever` interface without touching the date layer.
