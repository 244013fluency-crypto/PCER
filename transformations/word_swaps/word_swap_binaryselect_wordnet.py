# ! /usr/bin/nev python
# -*-coding:utf8-*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WordNet transformation used by the BinarySelect reproduction."""

from typing import Optional

from nltk.corpus import wordnet

import textattack
from .word_swap import WordSwap


class WordSwapBinarySelectWordNet(WordSwap):
    """Replace a word with synonyms returned by NLTK WordNet.

    The released BinarySelect notebook collects every ``lemma.name()`` and does
    not remove underscore-joined lemmas. Therefore
    ``allow_multiword_lemmas=True`` is the reproduction-oriented default.

    Args:
        language: WordNet language identifier. The paper's implementation uses
            English (``"eng"``).
        max_candidates: Optional cap on the number of returned synonyms. ``None``
            reproduces the uncapped notebook behavior.
        allow_multiword_lemmas: Whether underscore-joined WordNet lemmas are
            retained. Set to ``False`` for standard TextAttack-style one-word
            candidates.
        preserve_case: Whether candidate capitalization should follow the source
            word. The notebook does not recover case, so the default is ``False``.
    """

    def __init__(
        self,
        language: str = "eng",
        max_candidates: Optional[int] = None,
        allow_multiword_lemmas: bool = True,
        preserve_case: bool = False,
    ):
        super().__init__()

        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive or None.")

        try:
            wordnet.ensure_loaded()
        except LookupError as exc:
            raise LookupError(
                "NLTK WordNet is unavailable. Run "
                "`python -m nltk.downloader wordnet omw-1.4` before the attack."
            ) from exc

        if language not in wordnet.langs():
            raise ValueError(
                "Language {!r} is not available in NLTK WordNet. Available "
                "languages: {}".format(language, wordnet.langs())
            )

        self.language = language
        self.max_candidates = max_candidates
        self.allow_multiword_lemmas = allow_multiword_lemmas
        self.preserve_case = preserve_case

    def _get_replacement_words(self, word, random=False):
        del random  # Kept only for compatibility with some TextAttack versions.

        candidates = []
        seen = set()

        for synset in wordnet.synsets(word, lang=self.language):
            for candidate in synset.lemma_names(lang=self.language):
                if candidate == word:
                    continue

                if not self.allow_multiword_lemmas:
                    if "_" in candidate:
                        continue
                    if not textattack.shared.utils.is_one_word(candidate):
                        continue

                if self.preserve_case:
                    candidate = recover_word_case(candidate, word)

                if candidate in seen:
                    continue

                seen.add(candidate)
                candidates.append(candidate)

                if (
                    self.max_candidates is not None
                    and len(candidates) >= self.max_candidates
                ):
                    return candidates

        return candidates

    def extra_repr_keys(self):
        return [
            "language",
            "max_candidates",
            "allow_multiword_lemmas",
            "preserve_case",
        ]


def recover_word_case(word: str, reference_word: str) -> str:
    """Match lowercase, uppercase, or capitalized source-word casing."""

    if reference_word.islower():
        return word.lower()
    if reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    if (
        len(reference_word) > 1
        and reference_word[0].isupper()
        and reference_word[1:].islower()
    ):
        return word.capitalize()
    return word
