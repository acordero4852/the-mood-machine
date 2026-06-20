# Model Card: Mood Machine This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn I built and compared **both** versions on the same dataset.

## 1. Model Overview

**Model type:**  
Both. A hand-written rule based classifier (`mood_analyzer.py`) and a learned bag-of-words + logistic regression mode (`ml_experiments.py`), trained and evaluated on the same `SAMPLE_POSTS` / `TRUE_LABELS`.

**Intended purpose:**  
Classify short, social-media-style text into one of four moods: `positive`, `negative`, `neutral`, or `mixed`. It is a teaching project for understanding how modeling and data choices affect behavior — not for production use.

**How it works (brief):**  
- *Rule based:* tokenize the text, look each token up in positive/negative word lists (plus emoji and slang tables), add up a signed score, and map the score to a label. Negation flips the next word's sign.
- *ML:* `CountVectorizer` turns each post into word-count features; a `LogisticRegression` learns weights for those words from the labeled examples.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` now contains **26 posts** with 26 matching labels in `TRUE_LABELS` (the two lists are kept the same length). I started from the 6 starter posts and added two batches of 10. New posts were written to look like real messages and to deliberately stress the model with slang, emojis, sarcasm, and mixed feelings.

**Labeling process:**  
I labeled each post myself using the four allowed labels. The rule of thumb: `mixed` when a post clearly expresses both a positive and a negative feeling (e.g. "so happy and so tired at the same time"), `neutral` when there's little emotional content ("got a new plant today"), and positive/negative otherwise. Hard-to-label posts I expect reasonable people to disagree on:
- "meh, it was okay I guess" — I labeled it `neutral`, but it could read mildly negative.
- "had a good cry, feeling lighter now 🥲" — labeled `mixed`; the 🥲 emoji is genuinely ambiguous (happy-tears vs. sad).
- Sarcastic posts ("Oh great, another meeting…") — labeled by *intended* meaning (`negative`), not the literal words.

**Important characteristics of your dataset:**  
- Contains slang (`lowkey`, `highkey`, `no cap`, `fire`, `meh`, `ugh`)
- Contains emojis and emoticons (`🔥 😂 💀 🥲 :) :(`)
- Includes sarcasm (3 posts)
- Several posts express mixed feelings
- Several short/ambiguous messages

**Possible issues with the dataset:**  
- **Small** (26 posts) — far too small to train a model that generalizes.
- **Class imbalance** — fewer `mixed`/`neutral` than positive/negative.
- **Author bias** — written by one person in casual American English internet slang; both the language style and the "correct" labels reflect my own interpretation.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Implemented in `mood_analyzer.py` (`preprocess`, `_analyze`, `score_text`, `predict_label`):

- **Word scoring:** each token in `POSITIVE_WORDS` adds +1, each in `NEGATIVE_WORDS` subtracts 1. Every occurrence counts (not just presence).
- **Negation handling:** a negation word (`not`, `never`, `no`, `cant`, …) flips the sign of the token immediately after it, so "not happy" scores negative and "not bad" scores positive.
- **Weighted emoji/slang signals:** small lookup tables give stronger weights to emojis and slang, e.g. `🔥 😂 :)` → +2, `💀 :( ugh` → −2, `meh` → −1.
- **Preprocessing:** lowercasing, punctuation stripping, emoji separated out as their own tokens, and repeated-character normalization ("soooo" → "soo").
- **Label thresholds:** if a post has *both* positive and negative hits → `mixed`; otherwise score > 0 → `positive`, < 0 → `negative`, = 0 → `neutral`.

**Strengths of this approach:**  
- Fully **transparent / explainable** — `explain()` prints exactly which words drove the score.
- Predictable on clear, literal posts and on plain positive/negative wording.
- Negation and the `mixed` rule work well: "I am not happy about this" → `negative`, "not bad at all, actually kinda great" → `positive`, and "so happy and so tired" → `mixed` are all correct.

**Weaknesses of this approach:**  
The clearest failure mode is **sarcasm**. Because the model only adds up the
sentiment of individual words, a sarcastic sentence built from positive words is
read literally — the opposite of its true meaning. It also depends entirely on
the word lists: a real emotion word that isn't listed is invisible (e.g.
`relieved`, `nervous`, and `ready` are all unlisted, so those posts score 0 and
fall through to `neutral`). See §5 for specific misclassified examples.

## 4. How the ML Model Works (if used)

**Features used:**  
Bag of words via `CountVectorizer` — each post becomes a vector of word counts.
There is no negation, ordering, or emoji handling beyond what the tokenizer
keeps; the model only sees which words appear and how often.

**Training data:**  
Trained on the full `SAMPLE_POSTS` / `TRUE_LABELS` (26 labeled posts) using
`LogisticRegression`.

**Training behavior:**  
Because the model is evaluated on the **same** posts it trained on, it reached
**100% training accuracy** — it essentially memorized the dataset. Adding more
posts or relabeling existing ones directly changes what it memorizes; with a set
this small there is no held-out test set, so this number measures memorization,
**not** real-world accuracy.

**Strengths and weaknesses:**  
- *Strength:* learns word→label associations automatically, including the sarcastic posts the rule based model got wrong — without any hand-written rules.
- *Weakness:* almost certainly **overfit**. It "knows" that "traffic" or "taxes" means negative only because it saw those exact posts labeled negative. On a new sarcastic sentence with different words it would have no reason to get it right. It is also highly **sensitive to the labels** — flip one label and the learned weights for those words flip with it.

## 5. Evaluation

**How you evaluated the model:**  
Both models were run on all 26 labeled posts in `dataset.py` (`python main.py` and `python ml_experiments.py`).

- **Rule based accuracy: 0.73 (19/26).**
- **ML accuracy: 1.00 (26/26)** — but this is *training* accuracy on the same data, so it overstates real performance.

**Examples of correct predictions:**  
- "I am not happy about this" → `negative` (rule based). The negation rule flips `happy`, so the score goes negative — correct.
- "so happy and so tired at the same time 😂" → `mixed` (both). Both a positive signal (`happy`, `😂`) and a negative one (`tired`) are present.
- "Today was a terrible day" → `negative` (both). A single strong negative word with no competing signal.

**Examples of incorrect predictions (rule based):**  
- **Sarcasm — "I absolutely love getting stuck in traffic"** → predicted `positive`, true `negative`. `love` scores +1 and the model has no way to see the sentence is sarcastic.
- **Sarcasm — "Oh great, another meeting that could've been an email"** → predicted `positive`, true `negative`. `great` scores +1; the frustrated tone is invisible to a word-counting rule.
- **Sarcasm — "wow, cant wait to do my taxes"** → predicted `neutral`, true`negative`. None of these words are in the lists, so the score is 0.
- **Missing vocabulary — "just finished the assignment, feeling relieved"** → predicted `neutral`, true `positive`. `relieved` isn't in `POSITIVE_WORDS`.
- **Missing vocabulary — "lowkey nervous but highkey ready"** → predicted `neutral`, true `mixed`. `nervous` and `ready` are unlisted, so no signal fires.
- **Mixed read as one-sided — "I hate that this made me cry but here we are"** → predicted `negative`, true `mixed`. Only negative words register.
- **Weight sensitivity — "meh, it was okay I guess"** → predicted `negative`, true `neutral`. The slang `meh` (−1) outweighs otherwise neutral text.

**How their failures differed:**  
The ML model got **all seven** of the above correct on this dataset — but only
because it had already seen those exact sentences and their labels during
training. So the comparison is not "ML is smarter," it's "ML memorized the test."
The rule based model's errors are *predictable and explainable*; the ML model's
perfect score *hides* its real weakness (it would likely fail on unseen text).

## 6. Limitations

- The dataset is **small (26 posts)** and labeled by one person — not enough to train or fairly evaluate a learned model.
- The ML accuracy (1.00) is **training accuracy with no held-out test set**, so it measures memorization, not generalization.
- **The rule based model cannot detect sarcasm reliably** — sarcasm caused 3 of its 7 errors (e.g. "Oh great, another meeting…" → `positive` instead of `negative`). No simple scoring rule fixes this without overfitting to specific sentences.
- The rule based model is **only as good as its word lists** — unlisted emotion words (`relieved`, `nervous`) produce a score of 0 and a wrong `neutral`.
- Neither model handles longer or more contextual text; both look at words in near-isolation.

## 7. Ethical Considerations

- **Bias and scope:** the dataset is written in casual, American, internet-native English — slang like `lowkey`, `no cap`, `fire`, and emoji conventions. The model is effectively optimized for people who write that way. It would likely
- **misinterpret** other English dialects (e.g. AAVE, regional or non-US slang), formal writing, or non-native phrasing, and it has zero coverage of other languages. Sarcasm and cultural references that don't match the author's would also be misread.
- **Misclassifying distress:** reading a message expressing genuine distress as `neutral` or `positive` could matter a lot if such a system were used for, say, wellness or moderation. A model that misses sarcasm and unlisted emotion words should never be trusted to flag someone's emotional state.
- **Privacy:** mood detection on personal messages is sensitive; analyzing such text without consent raises clear privacy concerns.

## 8. Ideas for Improvement

- Add **much more labeled data**, and use a **held-out test set** (or cross-validation) so accuracy reflects generalization rather than memorization.
- Use **TF-IDF** instead of raw counts, and add features the bag of words misses (negation, emoji, n-grams).
- Expand the rule based word lists and emoji/slang tables; treat the gaps found in §5 (`relieved`, `nervous`, `ready`) as a starting list.
- Try a small **pretrained language model**, which can pick up sarcasm and context that neither current model can.
- Have **multiple people label** the data to reduce single-author bias and to measure how often humans disagree on the hard cases.
