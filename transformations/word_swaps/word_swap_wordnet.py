"""
Word Swap by swapping synonyms in WordNet
------------------------------------------------
"""


import nltk
from nltk.corpus import wordnet
import textattack

from .word_swap import WordSwap
import wn
wn.download("oewn:2024")
_OEWN=wn.Wordnet("oewn:2024")

class WordSwapWordNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet.

    >>> from textattack.transformations import WordSwapWordNet
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapWordNet()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, language="eng"):
        # nltk.download("omw-1.4")
        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language

    def _get_replacement_words(self, word, random=False):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        synonyms = set()
        for syn in wordnet.synsets(word, lang=self.language):
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (textattack.shared.utils.is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)

class WordSwapOMWEnglish(WordSwap):


    def __init__(self):

        self.language = "eng"

    def _get_replacement_words(self, word, random=False):
        synonyms = set()
        for syn in _OEWN.synsets(word):
            for w in syn.words():
                cand = w.lemma()
                if (
                    (cand != word)
                    and ("_" not in cand)
                    and (textattack.shared.utils.is_one_word(cand))
                ):
                    synonyms.add(cand)
        return list(synonyms)

