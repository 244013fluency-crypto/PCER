"""
Word Swap by generating visually similar words using Mistral-7B-Instruct
-----------------------------------------------------------------------
"""

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textattack
from .word_swap import WordSwap  # 继承 TextAttack 的 WordSwap 基类
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class WordSwapMistral(WordSwap):
    """Transforms an input by replacing its words with visually similar real words
    using a local Mistral-7B-Instruct model.

    >>> from textattack.transformations import WordSwapMistral
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapMistral()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I love nature.'
    >>> augmenter.augment(s)
    """

    def __init__(self, short_language_code):
        self.short_language_code = short_language_code  # e.g. 'cs' for Czech
        self.model_path = "/media/xmx/data/Mistral-7B-Instruct-v0.3/LLM-Research/Mistral-7B-Instruct-v0___3"
        if short_language_code == 'cs':
            self.language_code = "Czech"
        elif short_language_code == 'de':
            self.language_code = "German"
        elif short_language_code == 'en':
            self.language_code = "English"
        else:
            self.language_code = "Russian"

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 可改为目标 GPU 编号

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        self.generation_args = {
            "max_new_tokens": 200,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def _get_replacement_words(self, word, random=False):
        """Generate visually similar synonyms for the given word using the LLM."""
        prompt = (
            f"Provide 10 {self.language_code} words that are visually most similar to "
            f"[{word}] and must be real words. No repetition is allowed! "
            "The output should only contain similar words and no additional explanations or content!!!!!"
        )

        try:
            output = self.pipe(prompt, **self.generation_args)[0]['generated_text']
            return self._process_response(output, word)
        except Exception as e:
            print(f"[ERROR] Failed to get synonyms for '{word}': {e}")
            return []

    # def _process_response(self, s, original_word):
    #     """Cleans up model output using custom rules similar to the original script."""
    #     lines = [line.strip() for line in s.splitlines() if line.strip() and '*' not in line]
    #     # lines = [line.strip() for line in s.splitlines() if line.strip()]
    #     full_text = ' '.join(lines)
    #     text_no_digits = re.sub(r'[0-9\.]', ' ', full_text)
    #     words = [w for w in text_no_digits.split() if w and w.lower() != original_word.lower()]
    #     top_10 = words[:10]
    #     cleaned = [re.sub(r'[\(\（][^\)\）]*[\)\）]', '', w) for w in top_10]
    #     return cleaned

    def _process_response(self, s, original_word):
        """Cleans up model output using custom rules similar to the original script."""
        # 删除包含 * 的整行
        lines = [line.strip() for line in s.splitlines() if line.strip() and '*' not in line]

        # 合并为单个文本块
        full_text = ' '.join(lines)

        # 替换数字和点为空格
        text_no_digits = re.sub(r'[0-9\.]', ' ', full_text)

        # 分词并去除原始词
        words = [w for w in text_no_digits.split() if w and w.lower() != original_word.lower()]

        # 提取最多 10 个词
        top_10 = words[:10]

        # 如果不足 10 个，用空格填充
        if len(top_10) < 10:
            top_10 += [''] * (10 - len(top_10))

        # 删除括号及内容
        cleaned = [re.sub(r'[\(\（][^\)\）]*[\)\）]', '', w) for w in top_10]

        return cleaned

