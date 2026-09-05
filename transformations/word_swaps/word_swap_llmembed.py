import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModel
from .word_swap import WordSwap


class WordSwapNVEBot(WordSwap):
    """
    Transforms an input by replacing its words using the NV-Embed-v2 word embeddings and WordNet synonyms.

    Args:
        max_candidates (int): Maximum number of replacement candidates for each word.
        embedding_model_path (str): Path to the NV-Embed-v2 model.
    """

    def __init__(self, max_candidates=10, embedding_model_path='/media/xmx/data/nvembed', **kwargs):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates

        # Load embedding model and tokenizer
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(embedding_model_path, trust_remote_code=True).to(self.device)
        # self.model = AutoModel.from_pretrained(embedding_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

    def _get_replacement_words(self, word):
        """Get replacement candidates for a given word based on maximum cosine similarity."""
        try:
            # Step 1: Encode the input word
            # query_embedding = self.model.encode([word], max_length=512)
            query_embedding = self.model.encode(
                [word],
                instruction="Instruct: Find similar words\nQuery: ",
                max_length=128,
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

            # Step 2: Generate candidate words using WordNet
            candidate_words = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:  # Exclude the input word itself
                        candidate_words.add(lemma.name().replace("_", " "))  # Convert underscores to spaces

            for char in 'abcdefghijklmnopqrstuvwxyz':
                candidate_words.add(word + char)  # Append a character
                if len(word) > 1:
                    candidate_words.add(word[:-1])  # Remove last character
            candidate_words = list(candidate_words)
            if not candidate_words:
                return []  # If no candidates found, return an empty list
            if len(candidate_words) > self.max_candidates * 4:
                candidate_words = candidate_words[:self.max_candidates * 4]
            # candidate_words = candidate_words[:self.max_candidates * 4]
            # Step 3: Encode candidate words
            # candidate_embeddings = self.model.encode(candidate_words, max_length=512)
            candidate_embeddings = self.model.encode(
                candidate_words,
                instruction="Instruct: Find similar words\nCandidates: ",
                max_length=128,
            )
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)

            # Step 4: Compute cosine similarity
            similarities = (query_embedding @ candidate_embeddings.T).squeeze(0)

            # Step 5: Rank candidates by similarity (descending order)
            ranked_indices = torch.argsort(similarities, descending=True)

            # Step 6: Select top candidates based on max_candidates
            replacements = [candidate_words[idx] for idx in ranked_indices[: self.max_candidates]]

            return replacements
        except Exception as e:
            print(f"Error generating replacements for word '{word}': {e}")
            return []

    # def extra_repr_keys(self):
    #     return ["max_candidates", "embedding_model_path"]