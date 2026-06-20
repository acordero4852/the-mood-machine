"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# Additional posts covering slang, emojis, sarcasm, and mixed feelings.
# Each post below has a matching label appended to TRUE_LABELS so the two
# lists stay the same length.
SAMPLE_POSTS.extend([
    "Lowkey stressed but kind of proud of myself",      # mixed
    "This new album is fire no cap 🔥",                  # positive
    "I absolutely love getting stuck in traffic",       # negative (sarcasm)
    "ugh mondays 💀",                                    # negative
    "had a good cry, feeling lighter now 🥲",            # mixed
    "highkey obsessed with this show :)",               # positive
    "meh, it was okay I guess",                          # neutral
    "Oh great, another meeting that could've been an email",  # negative (sarcasm)
    "so happy and so tired at the same time 😂",         # mixed
    "nothing much happening today",                      # neutral
])

TRUE_LABELS.extend([
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "positive",  # "This new album is fire no cap 🔥"
    "negative",  # "I absolutely love getting stuck in traffic" (sarcasm)
    "negative",  # "ugh mondays 💀"
    "mixed",     # "had a good cry, feeling lighter now 🥲"
    "positive",  # "highkey obsessed with this show :)"
    "neutral",   # "meh, it was okay I guess"
    "negative",  # "Oh great, another meeting that could've been an email" (sarcasm)
    "mixed",     # "so happy and so tired at the same time 😂"
    "neutral",   # "nothing much happening today"
])

# A second batch of posts to grow the dataset further. Same rule applies:
# every new post here has exactly one matching label appended below.
SAMPLE_POSTS.extend([
    "best day ever, no cap 😂",                          # positive
    "I hate that this made me cry but here we are",      # mixed
    "wow, cant wait to do my taxes",                     # negative (sarcasm)
    "just finished the assignment, feeling relieved",    # positive
    "lowkey nervous but highkey ready",                  # mixed
    "the weather is fine I suppose",                      # neutral
    "this is literally the worst :(",                    # negative
    "not bad at all, actually kinda great",              # positive
    "tired of being stressed all the time",              # negative
    "got a new plant today",                              # neutral
])

TRUE_LABELS.extend([
    "positive",  # "best day ever, no cap 😂"
    "mixed",     # "I hate that this made me cry but here we are"
    "negative",  # "wow, cant wait to do my taxes" (sarcasm)
    "positive",  # "just finished the assignment, feeling relieved"
    "mixed",     # "lowkey nervous but highkey ready"
    "neutral",   # "the weather is fine I suppose"
    "negative",  # "this is literally the worst :("
    "positive",  # "not bad at all, actually kinda great" (negation)
    "negative",  # "tired of being stressed all the time"
    "neutral",   # "got a new plant today"
])

# Original TODO (now completed): Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
