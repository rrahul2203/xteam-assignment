# Support Triage and Answering, starter pack

Everything in this folder is yours to change, delete or replace. Nothing here is
sacred, including the baseline. Read `ML_Practical_Brief.pdf` for the task.

## What is in here

```
data/train.csv            400 labelled support messages (text,label)
                          routes: account-access, transaction-dispute, fraud-report, general

kb/                       31 support knowledge base documents, markdown with
                          YAML front matter. Every document carries:
                            doc_id, title, category, version,
                            effective_date, valid_until, status,
                            supersedes, superseded_by
                          Read the metadata. It is there for a reason.

questions.csv             38 customer questions (qid, question, as_of).
                          as_of is the date the question should be answered AS AT.

baseline/
  baseline_classifier.py  a working classifier handed over from a previous engineer
  eval_report.md          their evaluation and their recommendation to ship

media/                    optional, only if you take the stretch item
  screenshots/*.png       3 synthetic in-app screenshots
  voice/*.mp3             3 synthetic customer voice notes
```

All data, documents, screenshots and audio are synthetic and were generated for
this exercise. The product name in the assets is fictional. There is no real
customer data anywhere in this pack.

## Output formats we will run against

Your repo needs two entry points that read a CSV and write a CSV. Name the
modules and flags whatever you like, just document the exact commands in your
own README.

**1. Route classification**

```
<your predict command> --input <messages.csv> --output predictions.csv
```

`messages.csv` has a `text` column. `predictions.csv` must have `text` and
`prediction`.

**2. Question answering over the KB**

```
<your answer command> --input questions.csv --output answers.csv
```

`answers.csv` must have `qid` and `answer`, and should have `doc_ids` listing
the document or documents you used, semicolon separated. If your system chooses
not to answer, leave `answer` empty or say so in plain words.

We run both against a held back set, so build something that generalises.
