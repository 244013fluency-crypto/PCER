import nltk

# 设置 WordNet 数据目录
nltk.data.path.append("/media/xmx/data/prom_att2/oewn2024")
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModel
from .word_swap import WordSwap


def last_token_pool(last_hidden_states, attention_mask):
    """Extract the last token's hidden state based on attention mask."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description, query):
    """Combine task description with query to create a detailed input."""
    return f"<instruct>{task_description}\n<query>{query}"


# def get_text_embedding(texts, model, tokenizer, task=None, max_length=512):
#     """Generate embeddings for a list of texts using a pre-trained model."""
#     if task:
#         texts = [get_detailed_instruct(task, text) for text in texts]
#
#     batch_dict = tokenizer(
#         texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt", pad_to_multiple_of=8
#     )
#     with torch.no_grad():
#         outputs = model(**batch_dict)
#         embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
#     return F.normalize(embeddings, p=2, dim=1)


def get_text_embedding(texts, model, tokenizer, task=None, max_length=128):
    """Generate embeddings for a list of texts using a pre-trained model."""
    if task:
        texts = [get_detailed_instruct(task, text) for text in texts]

    batch_dict = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt", pad_to_multiple_of=8
    )

    # 确保输入和模型在同一设备
    device = next(model.parameters()).device
    batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


class WordSwapGEMMABOT(WordSwap):
    def __init__(self, model_path="/media/xmx/data/bge", max_candidates=10):
        """
        Args:
            model_path (str): Path to the pretrained BGE model.
            max_candidates (int): Maximum number of replacement words to return.
        """
        super().__init__()
        self.model_path = model_path
        self.max_candidates = max_candidates

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path).to("cpu")
        self.model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        # Explicitly set model to the second GPU
        # self.device = torch.device(
        #     "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = AutoModel.from_pretrained(model_path).to(self.device)
        # Check for GPU and enable multi-GPU support
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        # self.model = self.model.to(device)
        self.model.eval()

    def _get_replacement_words(self, word):
        """
        Generate replacement words for the given input word.

        Args:
            word (str): The input word to find replacements for.

        Returns:
            list: A list of the top `max_candidates` replacement words.
        """
        # Task description for embedding generation
        task = "Generate a meaningful embedding for a given text input to capture its semantic information."

        # Step 1: Get synonyms from WordNet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and "_" not in lemma.name():
                    synonyms.add(lemma.name().replace("_", " "))

        # Limit to max_candidates * 4
        synonyms = list(synonyms)[: self.max_candidates * 4]

        if not synonyms:
            return []

        # Step 2: Generate embeddings for the input word and synonyms
        texts = [word] + synonyms
        embeddings = get_text_embedding(texts, self.model, self.tokenizer, task=task)

        # Step 3: Compute cosine similarity
        word_embedding = embeddings[0]
        synonym_embeddings = embeddings[1:]
        similarities = torch.mv(synonym_embeddings, word_embedding)

        # Step 4: Select top `max_candidates` synonyms
        top_k = min(self.max_candidates, len(synonyms))  # Ensure top_k does not exceed available synonyms
        top_indices = torch.topk(similarities, top_k).indices
        top_synonyms = [synonyms[i] for i in top_indices.tolist()]

        return top_synonyms
