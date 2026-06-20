# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    # Emoticons / emoji treated as strong, standalone mood signals.
    # Value is the weight added to the score when the token appears.
    EMOJI_SCORES: Dict[str, int] = {
        ":)": 2,
        ":-)": 2,
        ":d": 2,    # ":D" lowercased by preprocess
        "🔥": 2,
        "😂": 2,
        ":(": -2,
        ":-(": -2,
        "💀": -2,
        # 🥲 is genuinely ambiguous ("happy tears" / bittersweet). We lean
        # slightly negative but it is a documented edge case.
        "🥲": -1,
    }

    # Slang words not in the starter dataset lists. Weight reflects intensity.
    SLANG_SCORES: Dict[str, int] = {
        "fire": 2,
        "obsessed": 2,
        "lit": 2,
        "meh": -1,
        "ugh": -2,
    }

    # Words that flip the sentiment of the word immediately after them.
    NEGATION_WORDS = {"not", "never", "no", "cant", "cannot", "dont", "wont"}

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        tokens: List[str] = []

        for raw in cleaned.split():
            # Keep known emoticons/emoji intact as their own signal tokens.
            if raw in self.EMOJI_SCORES:
                tokens.append(raw)
                continue

            # Pull any trailing emoji off the end of a word (e.g. "mondays💀").
            trailing = ""
            while raw and raw[-1] in self.EMOJI_SCORES:
                trailing = raw[-1] + trailing
                raw = raw[:-1]

            # Strip surrounding punctuation from the word itself, but leave
            # internal apostrophes alone so "don't" survives for negation.
            word = raw.strip(".,!?;:\"()[]{}")

            # Collapse 3+ repeated characters down to 2 ("soooo" -> "soo").
            word = re.sub(r"(.)\1{2,}", r"\1\1", word)

            if word:
                tokens.append(word)
            for emoji in trailing:
                tokens.append(emoji)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Core scoring routine shared by score_text, predict_label, and explain.

        Returns a tuple of:
          - score: the total numeric mood score
          - positive_hits: tokens that pushed the score up
          - negative_hits: tokens that pushed the score down

        Implements several improvements over naive presence counting:
          - Negation: a negation word ("not", "never", ...) flips the sentiment
            of the token immediately after it ("not happy" -> negative).
          - Counting: every occurrence contributes, not just the first.
          - Weights: emojis and slang carry stronger weights than plain words.
        """
        tokens = self.preprocess(text)

        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            # Determine this token's base sentiment weight.
            if token in self.positive_words:
                weight = 1
            elif token in self.negative_words:
                weight = -1
            elif token in self.EMOJI_SCORES:
                weight = self.EMOJI_SCORES[token]
            elif token in self.SLANG_SCORES:
                weight = self.SLANG_SCORES[token]
            else:
                continue

            # Look back one token for negation and flip the sentiment.
            prev = tokens[i - 1] if i > 0 else ""
            if prev in self.NEGATION_WORDS:
                weight = -weight

            score += weight
            if weight > 0:
                positive_hits.append(token)
            elif weight < 0:
                negative_hits.append(token)

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words and signals increase the score; negative ones decrease
        it. See _analyze for the modeling details (negation, counting, weights).
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        # If the text carries BOTH positive and negative signals, call it
        # "mixed" regardless of which side happens to win the score.
        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
