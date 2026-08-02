# Support Ticket Router, Evaluation Report

**Author:** previous engineer on the triage project
**Status:** handed over, awaiting sign off to ship

## Approach

Support messages arrive in a single queue and get routed by hand to one of four
teams: `account-access`, `transaction-dispute`, `fraud-report`, `general`. We
have 400 historical messages labelled with the route they were sent to.

I vectorised the messages with TF-IDF over unigrams and bigrams and trained a
logistic regression on top. I held out 20% of the data as a test set to check
performance.

Run it with:

```bash
python3 baseline/baseline_classifier.py
```

## Result

```
loaded 400 rows
test accuracy: 0.9875
```

**98.75% accuracy on held out data.** Only one message in the test set was
routed incorrectly.

## Class distribution

For reference, the label counts in the training data are:

| route | count |
|---|---|
| general | 160 |
| account-access | 100 |
| transaction-dispute | 90 |
| fraud-report | 50 |

`general` is the most common route and `fraud-report` the least common. The
model reaches 98.75% accuracy regardless, so the distribution does not appear to
be causing a problem in practice.

## Conclusion

The model is accurate, it trains in under a second, and it has no runtime
dependencies beyond scikit-learn. I see no blockers. Recommend we ship this to
production behind the existing queue and revisit only if accuracy degrades.

Possible follow ups, none of which I consider urgent:

- Try a gradient boosting model to see if it beats 98.75%.
- Consider an LLM if we ever add more routes.
