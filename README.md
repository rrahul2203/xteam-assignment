# ML practical

Three parts. Sections 1-2 are setup and tests, 3-5 are the commands, then one write-up per part —
each broken into one section per decision, so a section can be read on its own.

| Part | What it is | Headline | Write-up |
|---|---|---|---|
| **A** | Review of the handed-over route classifier | The 98.75% is unfalsifiable, not wrong | [Part A](#part-a--support-ticket-router-baseline-review) |
| **B** | Date-aware question answering over `kb/` | 0.933 doc accuracy, 0.000 wrong-answer rate | [Part B](#part-b--answering-from-the-knowledge-base) |
| **C** | Stretch: routing screenshots into A's four routes | 3/3 assets, 0 wrong-and-unflagged under blur | [Part C](#part-c--routing-screenshots) |

```
src/router/    Part A — data · crossval · model · evaluate · predict · tune · artifact
               Part C — screenshots · vlm_reader · screenshot_eval · screenshot_compare
src/qa/        Part B — kb · retriever · embeddings · answerer · eval_answers · answer_cli
               plus build_vectors · fetch_model (one-off setup)
tests/         test_kb · test_retriever · test_answerer · test_artifact · test_screenshots
eval/          Part B — gold.csv labels · doc_vectors.npz cached embeddings
models/        router.joblib (committed) · all-MiniLM-L6-v2 (fetched, gitignored)
notebook/      Part A — baseline_classifier_training · baseline_fix_walkthrough
```

**No LLM and no API key anywhere.** Part B is retrieval plus verbatim quoting; Part C is OCR plus
Part A's classifier. Everything runs locally.

---

# How to run it

**Verified from a clean `git clone` into an empty venv**, because "it works on my machine" is not a
claim a reviewer can check.

## 1. Setup — once

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt      # one file, includes pytest and torch
```

Two optional extras. Both are skipped rather than failed when absent:

```bash
# Part B: vendors the sentence-embedding model into models/ (~87MB, once).
# Skip it and Part B still runs, ranking lexically at 0.867 instead of 0.933.
./venv/bin/python -m src.qa.fetch_model

# Part C: the OCR engine. A system package, so pip cannot supply it.
brew install tesseract                          # or: apt install tesseract-ocr
```

## 2. Tests

```bash
./venv/bin/python -m pytest tests/ -q           # 33 passed
```

| Setup done | Result |
|---|---|
| Both | 33 passed |
| No `tesseract` | 30 passed, 3 skipped |
| No `fetch_model` | 31 passed, 2 skipped |
| Neither | 28 passed, 5 skipped |

The Part B skips need to encode a query; the Part C skips need OCR. Part C's redaction tests are
pure text and always run.

One test per behaviour, looping over inputs where a behaviour has more than one interesting case:

| File | Tests | Covers |
|---|---|---|
| `test_kb.py` | 5 | Window boundaries both ends, lapsing, malformed front matter, one version of a lineage in force per day, disjoint candidate pools |
| `test_answerer.py` | 7 | One question at two dates returning different documents, lapsed notices answering in the negative, abstention, verbatim answer text, rejected input |
| `test_retriever.py` | 7 | Fusion arithmetic and weight extremes, the vocabulary gap embeddings close, the lexical fallback, unit-norm vectors |
| `test_artifact.py` | 5 | Round trip identical in label and confidence, recorded provenance, prediction path loads or raises but never trains |
| `test_screenshots.py` | 9 | Redaction per identifier class and its ordering, two-pass recovery of low-contrast text, review gate on blurred, unreadable, and unscored reads |

## 3. Part A — route classification

```bash
# Batch, per the starter pack's output contract. Writes text,prediction.
./venv/bin/python -m src.router.predict --input messages.csv --output predictions.csv

# One message, straight to stdout.
./venv/bin/python -m src.router.predict --text "I see a transfer I never made"

# --confidence adds the top-class probability, which is what a review queue thresholds on.
./venv/bin/python -m src.router.predict --input messages.csv --confidence
```

```bash
./venv/bin/python -m src.router.evaluate        # the honest scores
./venv/bin/python -m src.router.tune            # re-derive tuned settings (slow, grid search)
./venv/bin/python -m src.router.artifact        # retrain and rewrite models/router.joblib
```

`predict` **loads** `models/router.joblib`; it does not fit. Pass `--retrain` to train in-process
instead, for development against edited data.

## 4. Part B — question answering

```bash
# Batch entry point. Writes qid,answer,doc_ids.
./venv/bin/python -m src.qa.answer_cli --input starter/questions.csv --output answers.csv

# One question, as at a date. The date is the whole point — try 2026-06-30 vs 2026-07-01.
./venv/bin/python -m src.qa.answer_cli --question "How long do I have to raise a dispute?" \
                                       --as-of 2026-03-01

./venv/bin/python -m src.qa.answer_cli --input starter/questions.csv --mode tfidf  # no embeddings
```

```bash
./venv/bin/python -m src.qa.eval_answers        # scores, strata, ablations, sweeps
./venv/bin/python -m src.qa.build_vectors       # refresh eval/doc_vectors.npz after editing kb/
```

As a library:

```python
from src.qa.answerer import AnswerService
service = AnswerService()                       # loads the KB and vectors once
result = service.answer("How long do I have to raise a dispute?", "2026-03-01")
result.text, result.doc_ids, result.status      # -> "...within 60 days...", ["kb-031"], "answered"
```

## 5. Part C — screenshot routing

```bash
# Routes every screenshot in a directory. Writes file,route,confidence,ocr_confidence,review,text.
./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots --output shots.csv

# One screenshot, straight to stdout.
./venv/bin/python -m src.router.screenshots --image starter/media/screenshots/txn-failed.png

./venv/bin/python -m src.router.screenshot_eval # blur sweep: how routing degrades
```

Needs `brew install tesseract` from step 1. Reuses `models/router.joblib`, so there is no second
model to build.

Extraction is pluggable, and the vision-model backend is selectable so the comparison in
[Part C](#part-c--routing-screenshots) is reproducible rather than asserted. It needs a local model
directory and is measurably worse:

```bash
./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots \
    --reader vlm --vlm-path /path/to/SmolVLM-500M-Instruct

# OCR against the vision model on the same scans. Runs the OCR tier alone without --vlm.
./venv/bin/python -m src.router.screenshot_compare --vlm /path/to/SmolVLM-500M-Instruct
```

---

# Part A — Support Ticket Router baseline review

Numbers below come from `notebook/baseline_classifier_training.ipynb`; commands are in
[How to run it](#3-part-a--route-classification). `crossval` holds the grouped-CV protocol that
`evaluate`, `model` and `tune` all score through, so the search and the reported number cannot drift
apart.

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

100%, which is worse news than 98.75%. I built the grouped split expecting it to break the model and
it didn't. Nothing in `train.csv` produces an error, so the set can't rank two models or catch a
regression. A 20-word vocabulary still scores 83.3%; 30 words, 84.8%. Mostly keyword lookup.

So accuracy is unmeasured: 98.75% isn't wrong, it's unfalsifiable. First deliverable is a few hundred
real messages labelled by someone who didn't write the model. Hand-written paraphrases scored 8/9 —
encouraging, and only 9 probes, so it stays an anecdote.

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

## The trained model is saved

`models/router.joblib` (228KB, committed) holds the fitted pipeline. `predict` loads it and **never
fits**. The reason is not speed on this data — training 400 rows takes 82ms — but that fitting is
proportional to the training set, so training inside the prediction path is a cost that grows with the
corpus and repeats on every call:

| Training rows | Fit | Load artifact |
|---|---|---|
| 400 (this repo) | 82ms | **7ms** |
| 4,000 | 385ms | 7ms |
| 16,000 | 1,354ms | 7ms |

Loading is flat; fitting is not. At a large corpus and high request volume the retraining version is
the wrong shape regardless of how fast it looks at n=400, so a missing artifact raises with the
command to build it rather than quietly training and hiding the cost.
`test_get_pipeline_never_calls_train` pins this by making `train` raise if the prediction path
touches it.

The second reason is provenance: the artifact stores the training-row count, a digest of `train.csv`,
the resolved hyperparameters (`C`, n-gram ranges, `FRAUD_SKEW`) and the scikit-learn version, so a
prediction traces to the model that made it. Edit `train.csv` and the next load warns that the digest
no longer matches — it still predicts, because refusing to load would break the CLI after any data
edit, but it will not do so silently. Training is deterministic (verified: identical coefficients
across fits), so the artifact is a convenience and an audit record rather than the only way to obtain
this model.

---

# Part B — Answering from the knowledge base

**No LLM and no API key.** Retrieval over 31 documents fusing two signals — TF-IDF and sentence
embeddings — with answers copied verbatim from the document cited. That is the grounding argument:
the answer text is a substring of the source, so it cannot drift from it, and `doc_ids` is checkable
by opening the file. `test_answer_text_is_copied_from_the_cited_document` asserts exactly this.

Embeddings are the brief's "embeddings and no LLM" option, and a required dependency, so the scores
below are what a reviewer reproduces by default. The service does not *break* without them: it ranks
lexically, says so in the log, and scores 0.867 instead of 0.933. Two tests pin that fallback by
simulating the dependency's absence. Commands are in
[How to run it](#4-part-b--question-answering).

**On the embedding model.** `fetch_model` keeps inference off the network: without a local copy the
first question of the first run pays the download inside the answering path; with one, `load_model()`
reads `models/all-MiniLM-L6-v2` **once per process** — 0.44s at startup, then a flat 28ms per
question. It skips files already present, retries a dropped connection three times, and writes to a
`.part` name so an interrupted download cannot look complete. The 87MB is gitignored, since weights
would sit in every clone's history forever
and one pinned command reproduces them, but `eval/doc_vectors.npz` (44KB) *is* committed so the
retrieval fixture travels with the repo.

## Retrieval: two signals, because they fail differently

Three rankings, scored on the same 38-row gold set. Every number here is from
`python3 -m src.qa.eval_answers`:

| | Doc accuracy | Wrong-answer | Abstention recall | Latency/question |
|---|---|---|---|---|
| TF-IDF | 0.867 | 0.053 | 0.875 | **12ms** |
| Embeddings | 0.900 | 0.026 | **1.000** | 29ms |
| **Hybrid, w=0.45** | **0.933** | **0.000** | 0.875 | 23ms |

Only four questions separate them, which is the whole argument in one table:

| qid | TF-IDF | Embeddings | Hybrid | What it turns on |
|---|---|---|---|---|
| q01 | ✗ wrong | ✓ | ✓ | "withdraw" vs the document's "transfers out" |
| q19 | ✗ wrong | ✓ | ✓ | two documents overlapping in wording |
| q24 | ✓ | ✗ wrong | ✓ | an exact term embeddings smear together |
| q38 | ✗ false | ✓ | ✗ false | unanswerable, but built from KB words |

**TF-IDF is precise and cheap.** Exact terms, figures and doc ids match on the token itself, so q24
lands on the right document; it needs no model, no 87MB of weights and no torch, and runs 2.4× faster
than the semantic path. Its ceiling is vocabulary — q01 and q19 are wrong because nothing in the
question shares words with the document that answers it, which no threshold fixes — and it carries
the worst wrong-answer rate of the three at 0.053.

**Embeddings generalise across wording.** They close both vocabulary gaps and are the only
configuration to decline every unanswerable question, 1.000 against 0.875: a question built from
in-KB words like q38 still lands far from every document in vector space, where TF-IDF sees overlap.
The cost is precision on exact terms (q24 regresses), a real dependency and 29ms.

**The hybrid wins because the two failure sets barely intersect.** It keeps q24 and gains q01 and
q19, reaching 0.933 document accuracy and the only 0.000 wrong-answer rate in the table — no
confidently-wrong answer on any of the 38 — at 23ms, less than embeddings alone since the lexical
pass is nearly free next to encoding. What it does not inherit is embeddings' clean abstention: q38
is still answered, hence 0.875.

Both signals are cosines, so they fuse as a weighted sum of raw values, deliberately not min-max
normalised per query — that would map the best candidate to 1.0 on every question and silently
disable the absolute abstention threshold.

Document vectors do not depend on `as_of`, so they are encoded once into `eval/doc_vectors.npz`
(44KB, committed) and date filtering is a row selection over them. Only the query is encoded per
call, which is why the cached fixture alone does not make the system semantic:
`Retriever.describe()` reports `tfidf` when the model is absent even though the document vectors are
present. Filter first, rank second still holds, and the date layer (`kb.py`) was not touched to add
any of this.

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

The measured finding is that **the obvious signals do not work**. Cosine cannot separate answerable
from unanswerable: being length-normalised, a question sharing two words with a short document scores
like one sharing eight with a long one, and the distributions overlap almost entirely (unanswerable
0.092–0.263 against answerable 0.101–0.477). The share of question words missing from the KB fails
too, so it is reported as a diagnostic and thresholded on nothing.

What separates them is *which* words are missing. Every unanswerable question contains a domain noun
absent from the whole corpus — margin, custodian, chargeback, electricity — while answerable ones are
missing only filler and dates. So abstention is a conjunction: a **long absent term** (≥6 chars,
absent from the KB) **and** low **salient coverage** (IDF-weighted share of the question's vocabulary
the top document contains, <0.35). Requiring both keeps "how many days did a customer have in March
2026" answerable while declining "what interest rate on a crypto backed margin loan", and moved
abstention recall 0.375 → 0.875.

A lapsed hit faces the same 0.35 bar without the absent-term escape, and otherwise yields to the best
in-force document — announcing an unrelated expiry is a confidently wrong answer, not a helpful one.

## How it was measured, and what it scores

`eval/gold.csv`: 38 hand-labelled rows carrying the expected `doc_ids`, a stratum, and for most
rows a regex the answer text must contain, so a right document quoting the wrong sentence still
fails.

| Metric | Score |
|---|---|
| Document accuracy (answerable, right doc cited) | **0.933** (28/30) |
| Wrong-answer rate (answered confidently, wrong doc) | **0.000** |
| Answer recall (answerable questions answered) | 0.933 |
| Abstention recall (unanswerable declined) | 0.875 |
| Fact accuracy (quoted text contains the expected value) | 0.731 |

Per stratum, with Wilson intervals — the normal approximation is useless at n=4:

| Stratum | Score | 95% CI |
|---|---|---|
| current | 21/22 = 0.955 | [0.78, 0.99] |
| historical | 3/4 = 0.750 | [0.30, 0.95] |
| lapsed | 4/4 = 1.000 | [0.51, 1.00] |
| unanswerable | 7/8 = 0.875 | [0.53, 0.98] |

One aggregate number would hide all four, and the strata fail for different reasons.

**Lexical against semantic against fused** is scored in
[Retrieval](#retrieval-two-signals-because-they-fail-differently) above — choosing embeddings was a
measurement, not a preference.

The weight sweep is flat at 0.933/0.000 across **0.38–0.52** and degrades either side, so 0.45 is the
centre of a plateau rather than a fitted point. `eval_answers.py` prints the sweep on a grid
deliberately tighter near the shipped weight, since a coarse step hides where the flat region ends.

**Where the hybrid loses** is abstention recall: 0.875 against a clean 1.000 for embeddings alone,
because semantic similarity gives q38 enough of a plausible neighbour to answer it. I kept the hybrid
because a 0.000 wrong-answer rate matters more than one extra abstention, but the trade is real and
the reverse of what I expected.

**Ablations, because they are what make the headline mean anything.** If switching the date filter
off does not move the score, the score is not measuring the thing this exercise is about:

| Configuration | Doc accuracy | Wrong-answer |
|---|---|---|
| Shipped | 0.933 | 0.000 |
| No date filter (rank whole KB) | 0.533 | 0.316 |
| No abstention (always answer top hit) | 0.967 | 0.026 |
| Coverage only, no absent-term conjunction | 0.767 | 0.000 |

Dropping the date filter costs 40% of document accuracy and takes the wrong-answer rate from zero to
0.316. That is the number saying the system does date resolution rather than keyword matching.

The second row is a cost, stated plainly: **abstaining loses document accuracy.** Answering
everything scores 0.967 against the shipped 0.933, because q12 and q22 are declined despite the right
document ranking first. Abstention buys a 0.000 wrong-answer rate and all 7 correct declines at the
price of those two — the trade I would defend for policy answers a customer acts on, but a trade and
not a free win. The last row is the other side: drop half the conjunction and accuracy falls to
0.767. `eval_answers.py` also prints a coverage sweep, so 0.35 is a point on a curve rather than an
assertion.

## The three failures

Named rather than tuned away — a stopping rule matters more here than 30/30 on n=38.

- **q12** — abstains on a March 2026 dispute question, just under the coverage bar. The date
  logic resolves correctly; the confidence gate is one notch too tight.
- **q22** — abstains on a Dogecoin deposit question that kb-102 does answer.
- **q38** — answers an unanswerable question whose vocabulary is entirely KB words. Embeddings
  made this one worse, not better.

Adding embeddings fixed q01 and q19 (both vocabulary gaps) and introduced none. Three things
were tested and **rejected**: min-max score normalisation before fusion (breaks the absolute
abstention threshold), a word/char feature weight sweep (flat across every weighting), and
sentence-level scoring instead of document-level (worse). Recorded rather than shipped.

---

# Part C — Routing screenshots

Stretch item. One modality, screenshots, routed into Part A's same four routes.

```bash
brew install tesseract                          # system package, pip cannot supply it

./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots
./venv/bin/python -m src.router.screenshots --image starter/media/screenshots/txn-failed.png
./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots --output shots.csv

./venv/bin/python -m src.router.screenshot_eval # the degradation table below
```

All three assets reach the right route, on redacted text, with no model beyond Part A's:

| Asset | Route | Route conf | OCR conf |
|---|---|---|---|
| `login-error.png` | `account-access` | 0.981 | 86.1 |
| `phishing-sms.png` | `fraud-report` | 0.515 | 94.3 |
| `txn-failed.png` | `transaction-dispute` | 0.912 | 94.6 |

Three files is a demonstration, not an evaluation, and nothing below is offered as an accuracy.

## Why screenshots, and why OCR

Screenshots, because their text is **rendered, not photographed**. The pixels came out of a font
renderer at a known size — no perspective, no motion blur, no background — which is the easy case
for OCR and the case where a vision model's extra capability buys nothing. Voice is harder on every
axis that matters here: it needs a real acoustic model rather than a system binary, it degrades on
accent and crosstalk in ways I cannot bound from three clips, and its failure mode is a plausible
wrong word rather than a visibly garbled one.

OCR rather than a vision model, for the same reason: the task is transcription, Tesseract is
already at 86–95 mean word confidence on these assets, and none of the four routes needs the layout
reasoning a vision model would earn its cost on. The route is decided by *what the customer wrote*,
which is text. So this reuses the Part A classifier unchanged — a screenshot and a typed ticket are
routed by the same model on the same four labels, and Part C adds an extraction stage rather than a
second classifier with its own drift and its own retraining story.

## OCR against a vision model, measured

`screenshot_compare.py` puts both approaches through the same 12 scans — 3 assets × 4 blur radii — routing with OCR plus the Part A classifier, then
with [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) (507M
parameters, local, on MPS) shown the image and asked for the route directly:

```bash
./venv/bin/python -m src.router.screenshot_compare --vlm /path/to/SmolVLM-500M-Instruct
```

| Tier | Correct | Latency, clean image | Routes used | r=0 | r=2 | r=3 | r=4 |
|---|---|---|---|---|---|---|---|
| OCR + Part A | 9/12 | 907–934ms | 4 of 4 | 3/3 | 3/3 | 2/3 | 1/3 |
| Vision LLM | 7/12 | 2368–2520ms | **2 of 4** | **1/3** | 2/3 | 2/3 | 2/3 |

The 7/12 is not a near miss, and the "routes used" column is why it is in the table. The model
answered `account-access` on **9 of the 12 scans** regardless of what was on screen, so its score is
one constant guess landing on the assets that happen to match it. Read its blur row left to right
and accuracy appears to *improve* as the image degrades — the signature of an answer that never
depended on the pixels.

Asking it to transcribe instead of route shows the same thing one level down:

| Asset | OCR | Vision LLM asked to transcribe |
|---|---|---|
| `login-error.png` | 99 words | *"Sign in page."* |
| `phishing-sms.png` | 136 words | *"Screen shows a message alert."* |
| `txn-failed.png` | 96 words | *"Transaction detail."* |

Those are captions, not transcriptions. At this size the model describes the *kind* of screen and
never reads a line of it, so it cannot see the `448120` admission that decides the fraud route —
and there is no intermediate output to inspect when it routes wrongly. OCR is 2.5× faster and
returns text a human can check.

## Either reader is selectable

**The choice is a flag, not a hardcoded assumption**, so the losing option stays runnable and the
claim above is falsifiable:

```bash
./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots   # --reader ocr default
./venv/bin/python -m src.router.screenshots --dir starter/media/screenshots \
    --reader vlm --vlm-path /path/to/SmolVLM-500M-Instruct
```

Both readers return `(text, confidence)` and feed the same classifier, so `--reader` changes how the
text is obtained and nothing else. Running the vision reader through the real pipeline is harsher
than the benchmark above, because the captions are what actually reach the classifier:
`login-error.png` → `account-access` ✓, `phishing-sms.png` → `transaction-dispute` ✗ (should be
`fraud-report`), `txn-failed.png` → `transaction-dispute` ✓. 2 of 3, and the miss is the fraud
ticket, routed to a billing queue on the strength of *"Screen shows a message alert."*

All three are held for review anyway, structurally: the vision reader reports **no per-word
confidence**, so the legibility gate has nothing to threshold. Rather than let an unscored read
inherit the pass a scored one gets, `route_image` holds every scan from a reader that cannot score
itself and the CSV leaves `ocr_confidence` empty. A backend with no confidence signal cannot run
unattended, which is a cost separate from its accuracy.

Scope: this rules out a **small local** VLM, the only tier measurable here. A hosted frontier model
would very likely transcribe these images correctly and there is no API access here to show
otherwise, so "a 500M local vision model cannot do this" is measured and "vision models cannot do
this" is untested. The argument against the hosted tier is not capability but
[where the data goes](#where-these-get-processed-and-why-that-is-the-whole-design): it sends customer
screenshots off the machine, and it bills per ticket where OCR does not.

## Two OCR passes, not one

These screenshots are light-on-dark, and one pass is not enough:

| Pass | `phishing-sms.png` words read | Blue bubble recovered |
|---|---|---|
| Default | 84 | no |
| Luminance-thresholded | 72 | **yes** |
| Union (shipped) | 136 | yes |

The customer's own message — *"I didn't make that withdrawal. The code is 448120"* — is white on
saturated blue, and it is the most important line in the ticket: the admission that they sent the
OTP. The default pass drops it **entirely**. Inverting does not recover it, because the bubble
collapses to mid-grey; a luminance threshold does. But the thresholded pass degrades faster on a
soft image (13 words vs 65 at blur 2), so neither pass alone wins and the union takes what either
one reads. Near-duplicate suppression across passes was tested and rejected: no route changed and
fraud confidence dropped slightly.

## What it costs and what it adds in latency

Measured on this machine, 8 cores, per screenshot at 780×1688:

| Stage | Time (median of 5, across the 3 assets) |
|---|---|
| OCR pass 1 (default) | 430–499 ms |
| OCR pass 2 (binarised) | 259–415 ms |
| Binarise | 7–11 ms |
| Classify redacted text | ~2 ms |
| **Per ticket, end to end** | **689–932 ms** |
| Model load | 7 ms, once per process |

**Marginal cost per ticket is zero** — no API call, no per-token or per-image billing, no egress,
just ~1 s of CPU on a box already running. At any volume the bill is capacity rather than usage,
which is the strongest argument for this over a hosted vision model.

The second pass adds 60–95% to the OCR time. That is the honest price of the blue bubble, and it
buys the one line that makes `phishing-sms.png` a fraud report. Both passes are independent and
would parallelise onto separate cores; I did not, because ~1 s already sits well inside the budget
for asynchronous triage — screenshots arrive attached to a ticket a human reads minutes later, and
the route needs to be right more than fast.

## When it degrades

Blurring each asset progressively and re-routing — `screenshot_eval` — turns this from a promise
into a table. 18 scans, 3 assets × 6 blur radii:

| Outcome | Count |
|---|---|
| Routed without review, correct | 12 |
| Routed without review, **wrong** | **0** |
| Held for review | 6 (3 of which would have been correct) |

Mean per-word OCR confidence gates the route at 60. Every scan that cleared the gate routed
correctly; all three misroutings fell below it. The failures are the interesting part: every
misroute lands on `general`, never on a wrong specific route. A soft screenshot loses its
distinguishing vocabulary first and keeps its generic words, so **degradation pulls toward the
harmless route rather than inventing a fraud report**. Route confidence falls alongside OCR
confidence (`login-error.png`: 0.981 at blur 0 → 0.485 at blur 4), so neither signal masks the
other. The drift is not monotonic — `phishing-sms.png` misroutes at blur 3 and lands back on
`fraud-report` at blur 4 — which is why the gate reads legibility instead of treating route
confidence as a proxy for it.

What the gate is *not* is a correctness predictor. OCR confidence overlaps between right and wrong
routes (correct scans span 22.7–94.6, wrong ones 37.0–43.0), so a low score does not mean the route
is wrong — 3 of the 6 held scans were right. It buys one narrow guarantee: no unreviewed route rests
on text nobody could read. The price is 3 needless reviews out of 18, which beats routing a fraud
report from a fragment.

Two other failures are handled separately. **No text at all** returns `review` with a null route and
never reaches the classifier, because an unreadable image is a failed read, not a `general` ticket.
**A dropped word** has no gate at all — a missing word lowers no confidence score — and the union of
two passes is the mitigation precisely because a word one pass loses the other may keep.

## Where these get processed, and why that is the whole design

The extracted text from three synthetic screenshots contains an email address, a phone number, a
6-digit OTP the customer admits sending, a Bitcoin address, a `$4,500.00` balance and a transaction
reference. A real queue would add card numbers and government IDs. **A screenshot is the
highest-risk attachment in support**, because the customer chose the crop, not you — they capture
the whole screen, including what you never asked for. That forces two decisions.

**Processing stays local.** OCR is a system binary and the classifier is a 228 KB local artifact, so
pixels and extracted text never cross a network boundary. A hosted vision API would put an OTP and a
wallet address in a third party's request logs — retained on their schedule, replicated to their
regions, inside their subprocessor list, possibly in a training corpus. That is a data-processing
agreement, a cross-border transfer question and a breach-notification surface, acquired to save ~1 s
of CPU on a task a local binary already does at 86–95 confidence. The cheaper option is also the one
that keeps the data in-house, so the recommendation does not depend on the cost argument holding.

**Identifiers are redacted before classification, not after.** Each pattern collapses to its
category name — `[otp]`, `[wallet]`, `[email]` — so the redaction is what the model sees and what
lands in the CSV, and no downstream store ever holds the value:

| Asset | Redacted | Route before | Route after |
|---|---|---|---|
| `login-error.png` | email | `account-access` 0.979 | `account-access` 0.981 |
| `phishing-sms.png` | wallet, phone, otp | `fraud-report` 0.517 | `fraud-report` 0.515 |
| `txn-failed.png` | — | `transaction-dispute` 0.912 | `transaction-dispute` 0.912 |

**Redaction is free.** No route changes and confidence moves by under 0.01, which is the point: the
route is decided by the customer's complaint, not by the digits in it. The category name is kept
deliberately — *that* an OTP was sent is the fraud signal; its value is only liability.

Ordering matters and is tested: `wallet` runs before `phone`, or the digit-run pattern eats the
middle of an all-digit address and leaves the `bc1q` prefix behind as literal text. And redaction is
text-only — **the source PNG still contains everything**. The routed text is clean; the attachment it
came from is not, so retention and access control on the stored image is the unsolved half and a real
deployment needs an image-retention policy to match.
