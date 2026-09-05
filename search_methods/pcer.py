#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PCER search method.

The default configuration is a plain WIR-guided best-first search using
WordNet substitutions. Four optional mechanisms are retained as independent
ablation switches and are disabled by default:

1. prior filtering + candidate-aware WIR;
2. stagnation-triggered semantic-neighborhood expansion;
3. nearest-checkpoint alternative-branch backtracking;
4. success-preserving post-attack restoration/refinement.

Readers can enable these mechanisms one at a time to examine their individual
effects while keeping the underlying search framework unchanged.
"""

from queue import PriorityQueue
import itertools
import re

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)
from textattack.transformations import WordSwapWordNet, WordSwapEmbedding


# ------------------------------------------------------------------
# Module-level candidate-space controller
# ------------------------------------------------------------------
# Some TextAttack versions do not expose the transformation object through the
# SearchMethod instance.  Therefore the search method switches candidate spaces
# through module-level state read by WordSwapWordNetEmbeddingSwitch.
_ACTIVE_CANDIDATE_SPACE = "wordnet"
_ACTIVE_EMBEDDING_TOP_K = None


def set_active_candidate_space(candidate_space="wordnet", embedding_top_k=None):
    global _ACTIVE_CANDIDATE_SPACE, _ACTIVE_EMBEDDING_TOP_K
    if candidate_space not in {"wordnet", "embedding"}:
        raise ValueError(
            f"Unsupported candidate_space {candidate_space}. Use 'wordnet' or 'embedding'."
        )
    _ACTIVE_CANDIDATE_SPACE = candidate_space
    _ACTIVE_EMBEDDING_TOP_K = embedding_top_k


class WordSwapWordNetEmbeddingSwitch(WordSwapWordNet):
    """Switchable WordNet/Counter-fitted-Embedding word substitution.

    WordNet is the default candidate space. The search method temporarily
    switches to the embedding space only when the optional stagnation-triggered
    semantic-neighborhood expansion mechanism requests it.
    """

    def __init__(self, embedding_max_candidates=50):
        super().__init__()
        self.wordnet_swap = WordSwapWordNet()
        self.embedding_swap = WordSwapEmbedding(max_candidates=embedding_max_candidates)
        self.embedding_max_candidates = embedding_max_candidates

    def _get_replacement_words(self, word):
        if _ACTIVE_CANDIDATE_SPACE == "embedding":
            replacement_words = self.embedding_swap._get_replacement_words(word)
            if _ACTIVE_EMBEDDING_TOP_K is not None and _ACTIVE_EMBEDDING_TOP_K > 0:
                replacement_words = replacement_words[: _ACTIVE_EMBEDDING_TOP_K]
            return replacement_words
        return self.wordnet_swap._get_replacement_words(word)

    def extra_repr_keys(self):
        return ["embedding_max_candidates"]


class PCERSearch(SearchMethod):
    """WIR-guided best-first search with four optional PCER mechanisms."""

    def __init__(
        self,
        # Optional mechanisms (all disabled by default for the plain baseline).
        enable_prior_candidate_aware=False,
        enable_stagnation_expansion=False,
        enable_checkpoint_backtracking=False,
        enable_success_restoration=False,
        # Base WIR / best-first search.
        wir_method="delete",
        unk_token="[UNK]",
        # Prior filtering + candidate-aware WIR.
        append_low_priority=False,
        prior_score_weight=0.05,
        candidate_score_weight=1.0,
        candidate_probe_top_k=3,
        first_round_probe_all=True,
        full_probe_every_n_modifications=2,
        base_wir_top_k=5,
        high_prior_reprobe_interval=2,
        exploration_rate=0.0,
        flip_bonus=1000.0,
        min_word_len=3,
        random_seed=None,
        # Shared improvement threshold used by expansion/backtracking.
        min_parent_gain=0.0,
        # Stagnation-triggered semantic-neighborhood expansion.
        stagnation_k1=2,
        stagnation_k2=4,
        embedding_top_k1=5,
        embedding_top_k2=15,
        rank_limit_k1=3,
        rank_limit_k2=5,
        # Success-preserving post-attack restoration/refinement.
        max_restore_attempts=None,
        max_refine_positions=None,
        max_refine_candidates=3,
        require_lexical_improvement=True,
        # Nearest-checkpoint alternative-branch backtracking.
        max_backtrack_uses=2,
        alternative_pool_size=32,
        local_gain_floor=0.0,
        backtrack_stagnation_patience=3,
    ):
        self.enable_prior_candidate_aware = enable_prior_candidate_aware
        self.enable_stagnation_expansion = enable_stagnation_expansion
        self.enable_checkpoint_backtracking = enable_checkpoint_backtracking
        self.enable_success_restoration = enable_success_restoration

        self.wir_method = wir_method
        self.unk_token = unk_token

        self.append_low_priority = append_low_priority
        self.prior_score_weight = prior_score_weight
        self.candidate_score_weight = candidate_score_weight
        self.candidate_probe_top_k = candidate_probe_top_k
        self.first_round_probe_all = first_round_probe_all
        self.full_probe_every_n_modifications = full_probe_every_n_modifications
        self.base_wir_top_k = base_wir_top_k
        self.high_prior_reprobe_interval = high_prior_reprobe_interval
        self.exploration_rate = exploration_rate
        self.flip_bonus = flip_bonus
        self.min_word_len = min_word_len
        self._rng = np.random.default_rng(random_seed)

        self.min_parent_gain = min_parent_gain

        self.stagnation_k1 = stagnation_k1
        self.stagnation_k2 = stagnation_k2
        self.embedding_top_k1 = embedding_top_k1
        self.embedding_top_k2 = embedding_top_k2
        self.rank_limit_k1 = rank_limit_k1
        self.rank_limit_k2 = rank_limit_k2

        self.max_restore_attempts = max_restore_attempts
        self.max_refine_positions = max_refine_positions
        self.max_refine_candidates = max_refine_candidates
        self.require_lexical_improvement = require_lexical_improvement
        self._restoration_candidate_provider = WordSwapWordNet()

        self.max_backtrack_uses = max_backtrack_uses
        self.alternative_pool_size = alternative_pool_size
        self.local_gain_floor = local_gain_floor
        self.backtrack_stagnation_patience = backtrack_stagnation_patience

        self._queue_counter = itertools.count()
        self._expanded_nodes = 0
        self._failed_expansions = set()

        # General stopwords used only in the sample part.
        self.stopwords = {
            "a", "about", "above", "across", "after", "again", "against",
            "all", "almost", "alone", "along", "already", "also", "although",
            "am", "among", "an", "and", "another", "any", "are", "as", "at",
            "be", "because", "been", "before", "being", "below", "between",
            "both", "but", "by", "can", "could", "did", "do", "does",
            "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here",
            "hers", "herself", "him", "himself", "his", "how", "i", "if",
            "in", "into", "is", "it", "its", "itself", "me", "more",
            "most", "my", "myself", "of", "off", "on", "once", "only",
            "or", "other", "our", "ours", "ourselves", "out", "over",
            "own", "same", "she", "should", "so", "some", "such", "than",
            "that", "the", "their", "theirs", "them", "themselves", "then",
            "there", "these", "they", "this", "those", "through", "to",
            "too", "under", "until", "up", "very", "was", "we", "were",
            "what", "when", "where", "which", "while", "who", "whom",
            "why", "with", "would", "you", "your", "yours", "yourself",
            "yourselves",
        }

        self.negation_words = {
            "no", "not", "never", "nothing", "neither", "nor", "without",
            "hardly", "barely", "scarcely",
        }

        self.intensity_words = {
            "very", "too", "really", "quite", "rather", "extremely",
            "highly", "completely", "totally", "absolutely", "almost",
            "still", "even", "less", "more",
        }

        self.sentiment_words = {
            "good", "great", "excellent", "wonderful", "nice", "amazing",
            "best", "better", "beautiful", "funny", "fun", "entertaining",
            "enjoyable", "enjoy", "enjoyed", "love", "loved", "lovely",
            "bad", "awful", "terrible", "poor", "boring", "worst", "worse",
            "dull", "weak", "stupid", "lazy", "waste", "hate", "hated",
            "disappointing", "ridiculous", "silly", "dumb", "mediocre",
            "mess", "masterpiece", "triumph", "clever", "smart", "precious",
            "sensitive", "complex", "powerful", "strong", "brilliant",
            "hilarious", "moving", "touching", "flat", "empty", "slow",
            "predictable", "unfunny", "annoying", "forgettable",
        }

        self.task_content_words = {
            "movie", "film", "story", "script", "plot", "scene", "scenes",
            "acting", "actor", "actors", "actress", "cast", "character",
            "characters", "performance", "performances", "director",
            "comedy", "drama", "horror", "music", "dance", "watch", "see",
            "seen", "make", "makes", "made", "recommend", "delivery",
            "message", "experience", "screen", "cinema", "dialogue",
            "ending", "role", "roles", "audience",
        }

        self.pronouns = {
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "them", "my", "your", "his", "their", "our", "its",
            "mine", "yours", "hers", "theirs", "ours", "myself", "yourself",
            "himself", "herself", "itself", "ourselves", "yourselves",
            "themselves",
        }

        self.determiners = {
            "a", "an", "the", "this", "that", "these", "those", "each",
            "every", "either", "neither",
        }

        self.auxiliaries = {
            "am", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had",
        }

        self.modals = {
            "can", "could", "may", "might", "must", "shall", "should",
            "will", "would",
        }

        self.prepositions = {
            "in", "on", "at", "to", "of", "for", "from", "by", "with",
            "as", "into", "onto", "over", "under", "through", "throughout",
            "across", "around", "between", "among", "about", "against",
            "before", "after", "above", "below", "behind", "beside",
            "within", "upon",
        }

        self.conjunctions = {
            "and", "or", "but", "so", "because", "if", "while", "although",
            "though", "unless", "until", "whereas", "nor",
        }

        self.wh_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        }

    @property
    def is_black_box(self):
        return self.wir_method != "gradient"

    def check_transformation_compatibility(self, transformation):
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    def extra_repr_keys(self):
        return [
            "enable_prior_candidate_aware",
            "enable_stagnation_expansion",
            "enable_checkpoint_backtracking",
            "enable_success_restoration",
            "wir_method",
            "append_low_priority",
            "prior_score_weight",
            "candidate_score_weight",
            "candidate_probe_top_k",
            "first_round_probe_all",
            "full_probe_every_n_modifications",
            "base_wir_top_k",
            "high_prior_reprobe_interval",
            "exploration_rate",
            "flip_bonus",
            "min_word_len",
            "min_parent_gain",
            "stagnation_k1",
            "stagnation_k2",
            "embedding_top_k1",
            "embedding_top_k2",
            "rank_limit_k1",
            "rank_limit_k2",
            "max_restore_attempts",
            "max_refine_positions",
            "max_refine_candidates",
            "require_lexical_improvement",
            "max_backtrack_uses",
            "alternative_pool_size",
            "local_gain_floor",
            "backtrack_stagnation_patience",
        ]

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def _normalize_word(self, word):
        return str(word).lower().strip()

    def _clean_word(self, word):
        return self._normalize_word(word).strip("[]").strip("\"'`“”‘’.,!?;:(){}")

    def _is_word_like(self, word):
        word = self._clean_word(word)
        return re.match(r"^[a-z][a-z'-]*$", word) is not None

    def _is_low_value_function_word(self, word):
        w = self._clean_word(word)
        protected_words = self.negation_words | self.intensity_words
        if w in protected_words:
            return False
        function_words = (
            self.pronouns
            | self.determiners
            | self.auxiliaries
            | self.modals
            | self.prepositions
            | self.conjunctions
            | self.wh_words
        )
        return w in function_words

    def _looks_like_content_word(self, word):
        w = self._clean_word(word)
        if len(w) < self.min_word_len:
            return False
        if not self._is_word_like(w):
            return False
        if self._is_low_value_function_word(w):
            return False
        if re.search(
            r"(ing|ed|ful|less|ous|ive|able|ible|al|ic|ly|ish|ary|ent|ant|ment|ness|tion|sion)$",
            w,
        ):
            return True
        if len(w) >= 4 and w not in self.stopwords:
            return True
        return False

    def _find_sample_start(self, words):
        lower_words = [self._clean_word(w).strip(":") for w in words]
        marker_patterns = [
            ["my", "first", "text", "is"],
            ["first", "text", "is"],
            ["text", "is"],
            ["sentence", "is"],
            ["review", "is"],
            ["input", "is"],
            ["input", "text", "is"],
        ]
        candidates = []
        for pattern in marker_patterns:
            plen = len(pattern)
            for i in range(len(lower_words) - plen + 1):
                if lower_words[i : i + plen] == pattern:
                    candidates.append(i + plen)
        if candidates:
            return max(candidates)
        for i, word in enumerate(words):
            raw = self._normalize_word(word)
            clean = self._clean_word(word)
            if raw.endswith(":") and clean in {"text", "sentence", "review", "input"}:
                return i + 1
        return 0

    def _prior_type_and_score(self, idx, word, sample_start):
        w = self._clean_word(word)
        if not self._is_word_like(w):
            return "excluded", 0.0
        if idx < sample_start:
            return "prompt", 1.0
        if len(w) < self.min_word_len:
            return "excluded", 0.0
        if w in self.sentiment_words:
            return "sentiment", 3.0
        if w in self.negation_words:
            return "negation", 3.0
        if w in self.intensity_words:
            return "intensity", 2.5
        if w in self.task_content_words:
            return "task_content", 2.5
        if self._is_low_value_function_word(w):
            return "excluded", 0.0
        if w in self.stopwords:
            return "excluded", 0.0
        if self._looks_like_content_word(w):
            return "general_content", 1.0
        return "excluded", 0.0

    def _split_indices_by_prior(self, initial_text, indices_to_order):
        words = initial_text.words
        sample_start = self._find_sample_start(words)
        high_priority_indices = []
        low_priority_indices = []
        prior_scores = {}
        prior_types = {}
        for idx in indices_to_order:
            if idx >= len(words):
                continue
            word = words[idx]
            ptype, score = self._prior_type_and_score(idx, word, sample_start)
            prior_scores[idx] = score
            prior_types[idx] = ptype
            if score > 0:
                high_priority_indices.append(idx)
            else:
                low_priority_indices.append(idx)
        return high_priority_indices, low_priority_indices, prior_scores, prior_types

    def _should_probe_candidates(
        self,
        idx,
        prior_type,
        base_rank,
        current_node_id,
        modification_count=0,
    ):
        if self.first_round_probe_all and current_node_id == 1:
            return True
        if (
            self.full_probe_every_n_modifications is not None
            and self.full_probe_every_n_modifications > 0
            and modification_count > 0
            and modification_count % self.full_probe_every_n_modifications == 0
        ):
            return True
        if self.base_wir_top_k is not None and self.base_wir_top_k > 0:
            if base_rank < self.base_wir_top_k:
                return True
        high_prior_types = {"sentiment", "negation", "intensity", "task_content"}
        if prior_type in high_prior_types and self.high_prior_reprobe_interval > 0:
            if current_node_id % self.high_prior_reprobe_interval == 0:
                return True
        if self.exploration_rate > 0 and self._rng.random() < self.exploration_rate:
            return True
        return False

    def _select_probe_candidates(self, candidates):
        if not candidates:
            return []
        if self.candidate_probe_top_k is None or self.candidate_probe_top_k <= 0:
            return candidates
        return candidates[: self.candidate_probe_top_k]

    def _candidate_aware_scores(
        self,
        current_result,
        indices_to_evaluate,
        base_wir_scores,
        prior_types,
        current_node_id,
        modification_count=0,
    ):
        initial_text = current_result.attacked_text
        current_score = current_result.score
        candidate_scores = np.zeros(len(indices_to_evaluate), dtype=float)
        candidate_cache = {}
        immediate_success_result = None
        search_over = False

        base_order_positions = np.argsort(-base_wir_scores)
        base_rank_map = {
            int(indices_to_evaluate[pos]): int(rank)
            for rank, pos in enumerate(base_order_positions)
        }

        for pos, idx in enumerate(indices_to_evaluate):
            idx = int(idx)
            should_probe = self._should_probe_candidates(
                idx=idx,
                prior_type=prior_types.get(idx, "unknown"),
                base_rank=base_rank_map.get(idx, len(indices_to_evaluate)),
                current_node_id=current_node_id,
                modification_count=modification_count,
            )
            if not should_probe:
                candidate_cache[idx] = {
                    "texts": [],
                    "results": [],
                    "evaluated_all": False,
                    "skipped_by_gate": True,
                }
                continue

            set_active_candidate_space("wordnet", None)
            transformed_text_candidates = self.get_transformations(
                initial_text,
                original_text=initial_text,
                indices_to_modify=[idx],
            )
            if not transformed_text_candidates:
                candidate_cache[idx] = {
                    "texts": [],
                    "results": [],
                    "evaluated_all": True,
                    "skipped_by_gate": False,
                }
                continue

            probe_candidates = self._select_probe_candidates(transformed_text_candidates)
            if not probe_candidates:
                candidate_cache[idx] = {
                    "texts": [],
                    "results": [],
                    "evaluated_all": False,
                    "skipped_by_gate": False,
                }
                continue

            probe_results, search_over = self.get_goal_results(probe_candidates)
            candidate_cache[idx] = {
                "texts": probe_candidates,
                "results": probe_results,
                "evaluated_all": len(probe_candidates) == len(transformed_text_candidates),
                "skipped_by_gate": False,
            }
            if not probe_results:
                if search_over:
                    break
                continue
            best_result = max(probe_results, key=lambda r: r.score)
            best_gain = max(best_result.score - current_score, 0.0)
            candidate_scores[pos] = best_gain
            for result in probe_results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    immediate_success_result = result
                    candidate_scores[pos] = best_gain + self.flip_bonus
                    set_active_candidate_space("wordnet", None)
                    return candidate_scores, candidate_cache, immediate_success_result, search_over
            if search_over:
                break
        set_active_candidate_space("wordnet", None)
        return candidate_scores, candidate_cache, immediate_success_result, search_over

    def _compute_base_wir_scores(self, current_result, indices_to_evaluate):
        initial_text = current_result.attacked_text
        set_active_candidate_space("wordnet", None)
        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_evaluate
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i)
                for i in indices_to_evaluate
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        elif self.wir_method == "weighted-saliency":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_evaluate
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])
            softmax_saliency_scores = softmax(torch.Tensor(saliency_scores), dim=0).numpy()
            delta_ps = []
            for idx in indices_to_evaluate:
                if search_over:
                    delta_ps = delta_ps + [0.0] * (len(softmax_saliency_scores) - len(delta_ps))
                    break
                set_active_candidate_space("wordnet", None)
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                delta_ps.append(np.max(score_change))
            index_scores = softmax_saliency_scores * np.array(delta_ps)
        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len(indices_to_evaluate))
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, index in enumerate(indices_to_evaluate):
                matched_tokens = word2token_mapping[index]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)
            search_over = False
        elif self.wir_method == "random":
            index_scores = np.random.rand(len(indices_to_evaluate))
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")
        return index_scores, search_over

    def _count_modified_words(self, current_text, reference_text):
        current_words = list(current_text.words)
        reference_words = list(reference_text.words)
        max_len = max(len(current_words), len(reference_words))
        modified = 0
        for i in range(max_len):
            cur = current_words[i] if i < len(current_words) else None
            ref = reference_words[i] if i < len(reference_words) else None
            if cur != ref:
                modified += 1
        return modified

    def _modified_indices(self, current_text, reference_text):
        current_words = list(current_text.words)
        reference_words = list(reference_text.words)
        max_len = max(len(current_words), len(reference_words))
        modified = set()
        for i in range(max_len):
            cur = current_words[i] if i < len(current_words) else None
            ref = reference_words[i] if i < len(reference_words) else None
            if cur != ref:
                modified.add(i)
        return modified

    def _get_index_order_and_cache(self, current_result, max_len=-1, modification_count=0):
        initial_text = current_result.attacked_text
        _, indices_to_order = self.get_indices_to_order(initial_text)
        if indices_to_order is None or len(indices_to_order) == 0:
            return np.array([], dtype=int), False, {}, None
        if max_len > 0:
            indices_to_order = indices_to_order[:max_len]

        # Plain baseline: standard WIR ranking over all eligible indices.
        if not self.enable_prior_candidate_aware:
            indices_to_evaluate = list(indices_to_order)
            base_wir_scores, search_over = self._compute_base_wir_scores(
                current_result, indices_to_evaluate
            )
            index_order = np.array(indices_to_evaluate)[(-base_wir_scores).argsort()]
            return index_order, search_over, {}, None

        # Optional mechanism 1: prior filtering + candidate-aware WIR.
        high_priority_indices, low_priority_indices, prior_scores, prior_types = (
            self._split_indices_by_prior(initial_text, indices_to_order)
        )
        if not high_priority_indices:
            high_priority_indices = list(indices_to_order)
            low_priority_indices = []
            prior_scores = {int(idx): 1.0 for idx in high_priority_indices}
            prior_types = {int(idx): "fallback" for idx in high_priority_indices}

        indices_to_evaluate = high_priority_indices
        base_wir_scores, search_over = self._compute_base_wir_scores(
            current_result, indices_to_evaluate
        )
        if search_over:
            index_order = np.array(indices_to_evaluate)[(-base_wir_scores).argsort()]
            return index_order, search_over, {}, None

        candidate_scores, candidate_cache, immediate_success_result, search_over = (
            self._candidate_aware_scores(
                current_result=current_result,
                indices_to_evaluate=indices_to_evaluate,
                base_wir_scores=base_wir_scores,
                prior_types=prior_types,
                current_node_id=self._expanded_nodes,
                modification_count=modification_count,
            )
        )
        prior_bias = np.zeros(len(indices_to_evaluate), dtype=float)
        if self.prior_score_weight > 0:
            prior_bias = np.array(
                [prior_scores.get(idx, 0.0) for idx in indices_to_evaluate]
            )
        final_scores = (
            base_wir_scores
            + self.prior_score_weight * prior_bias
            + self.candidate_score_weight * candidate_scores
        )
        index_order = np.array(indices_to_evaluate)[(-final_scores).argsort()]
        if self.append_low_priority and low_priority_indices:
            index_order = np.concatenate(
                [index_order, np.array(low_priority_indices, dtype=int)]
            )
        return index_order, search_over, candidate_cache, immediate_success_result

    # ------------------------------------------------------------------
    # Candidate evaluation and expansion policy
    # ------------------------------------------------------------------
    def _evaluate_index_candidates_with_cache(
        self,
        current_result,
        initial_result,
        idx,
        candidate_cache,
        candidate_space="wordnet",
        embedding_top_k=None,
    ):
        current_text = current_result.attacked_text
        idx = int(idx)
        all_results = []
        search_over = False
        cached_text_keys = set()
        if candidate_space == "wordnet":
            cached = candidate_cache.get(idx, None)
            if cached is not None:
                all_results.extend(cached.get("results", []))
                if cached.get("evaluated_all", False):
                    return all_results, False
                cached_text_keys = {str(text) for text in cached.get("texts", [])}
        set_active_candidate_space(candidate_space, embedding_top_k)
        transformed_text_candidates = self.get_transformations(
            current_text,
            original_text=initial_result.attacked_text,
            indices_to_modify=[idx],
        )
        set_active_candidate_space("wordnet", None)
        if not transformed_text_candidates:
            return all_results, False
        remaining_candidates = [text for text in transformed_text_candidates if str(text) not in cached_text_keys]
        if not remaining_candidates:
            return all_results, False
        remaining_results, search_over = self.get_goal_results(remaining_candidates)
        all_results.extend(remaining_results)
        return all_results, search_over

    def _embedding_policy(self, rank, semantic_no_improve_steps):
        if semantic_no_improve_steps >= self.stagnation_k2 and rank < self.rank_limit_k2:
            return self.embedding_top_k2
        if semantic_no_improve_steps >= self.stagnation_k1 and rank < self.rank_limit_k1:
            return self.embedding_top_k1
        return None

    # ------------------------------------------------------------------
    # Success-preserving post-attack restoration/refinement
    # ------------------------------------------------------------------
    def _is_success_result(self, result):
        return result.goal_status == GoalFunctionResultStatus.SUCCEEDED

    def _safe_restore_text(self, attacked_text, reference_text, idx):
        reference_words = list(reference_text.words)
        current_words = list(attacked_text.words)
        if idx < 0 or idx >= len(reference_words) or idx >= len(current_words):
            return None
        if current_words[idx] == reference_words[idx]:
            return None
        return attacked_text.replace_word_at_index(idx, reference_words[idx])

    def _levenshtein_distance(self, a, b):
        a, b = str(a), str(b)
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = cur
        return prev[-1]

    def _success_quality_key(self, result, initial_result):
        mod_count = self._count_modified_words(result.attacked_text, initial_result.attacked_text)
        return (-mod_count, result.score)

    def _candidate_texts_for_failed_restore(self, current_result, initial_result, idx):
        current_words = list(current_result.attacked_text.words)
        original_words = list(initial_result.attacked_text.words)
        if idx < 0 or idx >= len(current_words) or idx >= len(original_words):
            return []
        original_word = original_words[idx]
        current_word = current_words[idx]
        if current_word == original_word:
            return []
        try:
            candidates = list(self._restoration_candidate_provider._get_replacement_words(original_word))
        except Exception:
            candidates = []
        candidates = [w for w in candidates if w != original_word and w != current_word]
        current_dist = self._levenshtein_distance(current_word, original_word)
        selected = []
        for cand in candidates:
            if self.require_lexical_improvement:
                if self._levenshtein_distance(cand, original_word) >= current_dist:
                    continue
            selected.append(cand)
            if self.max_refine_candidates is not None and self.max_refine_candidates > 0:
                if len(selected) >= self.max_refine_candidates:
                    break
        return [current_result.attacked_text.replace_word_at_index(idx, cand) for cand in selected]

    def _postprocess_success(self, success_result, initial_result):
        if not self.enable_success_restoration:
            return success_result
        if success_result is None or not self._is_success_result(success_result):
            return success_result
        current_result = success_result
        modified = list(self._modified_indices(current_result.attacked_text, initial_result.attacked_text))
        modified = list(reversed(modified))
        if self.max_restore_attempts is not None and self.max_restore_attempts > 0:
            modified = modified[: self.max_restore_attempts]
        failed_restore_indices = []
        for idx in modified:
            restored_text = self._safe_restore_text(current_result.attacked_text, initial_result.attacked_text, idx)
            if restored_text is None:
                continue
            restored_results, search_over = self.get_goal_results([restored_text])
            if restored_results:
                restored_result = restored_results[0]
                if self._is_success_result(restored_result):
                    current_result = restored_result
                else:
                    failed_restore_indices.append(idx)
            else:
                failed_restore_indices.append(idx)
            if search_over:
                break
        if self.max_refine_positions is not None and self.max_refine_positions > 0:
            failed_restore_indices = failed_restore_indices[: self.max_refine_positions]
        for idx in failed_restore_indices:
            candidate_texts = self._candidate_texts_for_failed_restore(current_result, initial_result, idx)
            if not candidate_texts:
                continue
            candidate_results, search_over = self.get_goal_results(candidate_texts)
            success_candidates = [r for r in candidate_results if self._is_success_result(r)]
            if success_candidates:
                current_result = max(success_candidates, key=lambda r: self._success_quality_key(r, initial_result))
            if search_over:
                break
        return current_result

    # ------------------------------------------------------------------
    # Nearest-checkpoint alternative-branch recovery
    # ------------------------------------------------------------------
    def _add_checkpoint(self, checkpoints, current_result, alternatives, initial_result):
        if not self.enable_checkpoint_backtracking:
            return
        if not alternatives:
            return
        alternatives = sorted(alternatives, key=lambda r: r.score, reverse=True)
        if self.alternative_pool_size is not None and self.alternative_pool_size > 0:
            alternatives = alternatives[: self.alternative_pool_size]
        checkpoint = {
            "result": current_result,
            "alternatives": alternatives,
            "score": current_result.score,
            "mod_count": self._count_modified_words(current_result.attacked_text, initial_result.attacked_text),
            "order": next(self._queue_counter),
        }
        checkpoints.append(checkpoint)
        if self.alternative_pool_size is not None and self.alternative_pool_size > 0:
            del checkpoints[:-self.alternative_pool_size]

    def _activate_backtrack(self, checkpoints, priority_queue, used_backtrack_texts):
        if not self.enable_checkpoint_backtracking:
            return False
        while checkpoints:
            # Use the nearest checkpoint: the latest local decision point is
            # the first recovery target.
            checkpoint = checkpoints[-1]
            while checkpoint["alternatives"]:
                alt = checkpoint["alternatives"].pop(0)
                key = str(alt.attacked_text)
                if key in used_backtrack_texts:
                    continue
                used_backtrack_texts.add(key)
                priority_queue.put((-alt.score, next(self._queue_counter), alt))
                return True
            checkpoints.pop()
        return False

    # ------------------------------------------------------------------
    # Best-first search
    # ------------------------------------------------------------------
    def _process_results_for_state(
        self,
        results,
        current_result,
        initial_result,
        best_result,
        priority_queue,
        searched_texts,
        alternatives,
    ):
        search_best_child_gain = None
        for result in results:
            text_key = str(result.attacked_text)
            if text_key in searched_texts:
                continue
            searched_texts.add(text_key)
            search_best_child_gain = max(
                search_best_child_gain if search_best_child_gain is not None else float("-inf"),
                result.score - current_result.score,
            )
            if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                return best_result, self._postprocess_success(result, initial_result), search_best_child_gain
            if result.score > best_result.score:
                best_result = result
                priority_queue.put((-result.score, next(self._queue_counter), result))
            elif (
                self.enable_checkpoint_backtracking
                and result.score > current_result.score + self.local_gain_floor
            ):
                # These locally improving but globally non-admitted states are
                # not in the normal priority queue.  They become available only
                # when the nearest-checkpoint recovery mechanism is triggered.
                alternatives.append(result)
        return best_result, None, search_best_child_gain

    def perform_search(self, initial_result):
        self._expanded_nodes = 0
        self._queue_counter = itertools.count()
        self._failed_expansions = set()
        set_active_candidate_space("wordnet", None)

        priority_queue = PriorityQueue()
        priority_queue.put(
            (-initial_result.score, next(self._queue_counter), initial_result)
        )

        best_result = initial_result
        searched_texts = {str(initial_result.attacked_text)}
        checkpoints = []
        used_backtrack_texts = set()
        backtrack_uses = 0
        backtrack_no_improve_steps = 0
        semantic_no_improve_steps = 0

        while True:
            if priority_queue.empty():
                if (
                    self.enable_checkpoint_backtracking
                    and backtrack_uses < self.max_backtrack_uses
                ):
                    activated = self._activate_backtrack(
                        checkpoints, priority_queue, used_backtrack_texts
                    )
                    if activated:
                        backtrack_uses += 1
                    else:
                        set_active_candidate_space("wordnet", None)
                        return best_result
                else:
                    set_active_candidate_space("wordnet", None)
                    return best_result

            _, _, current_result = priority_queue.get()
            if current_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                set_active_candidate_space("wordnet", None)
                return self._postprocess_success(current_result, initial_result)

            previous_best_score = best_result.score
            self._expanded_nodes += 1
            modification_count = self._count_modified_words(
                current_result.attacked_text, initial_result.attacked_text
            )
            modified_indices = self._modified_indices(
                current_result.attacked_text, initial_result.attacked_text
            )

            set_active_candidate_space("wordnet", None)
            (
                index_order,
                search_over,
                candidate_cache,
                immediate_success_result,
            ) = self._get_index_order_and_cache(
                current_result, modification_count=modification_count
            )
            if immediate_success_result is not None:
                set_active_candidate_space("wordnet", None)
                return self._postprocess_success(
                    immediate_success_result, initial_result
                )
            if search_over:
                set_active_candidate_space("wordnet", None)
                return best_result

            current_text_key = str(current_result.attacked_text)
            alternatives = []
            node_best_child_gain = None

            for rank, idx in enumerate(index_order):
                idx = int(idx)

                # When neighborhood expansion is enabled, do not expand a position
                # that has already been modified on the current path.
                if (
                    self.enable_stagnation_expansion
                    and idx in modified_indices
                ):
                    continue

                # Plain WordNet expansion used by the baseline and all variants.
                wordnet_results, wordnet_search_over = (
                    self._evaluate_index_candidates_with_cache(
                        current_result=current_result,
                        initial_result=initial_result,
                        idx=idx,
                        candidate_cache=candidate_cache,
                        candidate_space="wordnet",
                        embedding_top_k=None,
                    )
                )
                best_result, success_result, child_gain = (
                    self._process_results_for_state(
                        wordnet_results,
                        current_result,
                        initial_result,
                        best_result,
                        priority_queue,
                        searched_texts,
                        alternatives,
                    )
                )
                if child_gain is not None:
                    node_best_child_gain = max(
                        node_best_child_gain
                        if node_best_child_gain is not None
                        else float("-inf"),
                        child_gain,
                    )
                if success_result is not None:
                    set_active_candidate_space("wordnet", None)
                    return success_result
                if wordnet_search_over:
                    set_active_candidate_space("wordnet", None)
                    return best_result

                # Optional mechanism 2: after semantic stagnation, expand a small
                # Counter-fitted embedding neighborhood for high-ranked positions.
                if self.enable_stagnation_expansion:
                    embedding_top_k = self._embedding_policy(
                        rank, semantic_no_improve_steps
                    )
                    if embedding_top_k is not None:
                        embedding_key = (
                            "embedding",
                            current_text_key,
                            idx,
                            embedding_top_k,
                        )
                        if embedding_key not in self._failed_expansions:
                            embedding_results, embedding_search_over = (
                                self._evaluate_index_candidates_with_cache(
                                    current_result=current_result,
                                    initial_result=initial_result,
                                    idx=idx,
                                    candidate_cache={},
                                    candidate_space="embedding",
                                    embedding_top_k=embedding_top_k,
                                )
                            )
                            best_result, success_result, emb_child_gain = (
                                self._process_results_for_state(
                                    embedding_results,
                                    current_result,
                                    initial_result,
                                    best_result,
                                    priority_queue,
                                    searched_texts,
                                    alternatives,
                                )
                            )
                            if emb_child_gain is not None:
                                node_best_child_gain = max(
                                    node_best_child_gain
                                    if node_best_child_gain is not None
                                    else float("-inf"),
                                    emb_child_gain,
                                )
                            if success_result is not None:
                                set_active_candidate_space("wordnet", None)
                                return success_result
                            if not embedding_results or not any(
                                r.score
                                > current_result.score + self.min_parent_gain
                                for r in embedding_results
                            ):
                                self._failed_expansions.add(embedding_key)
                            if embedding_search_over:
                                set_active_candidate_space("wordnet", None)
                                return best_result

            # Optional mechanism 3: retain non-admitted local improvements at the
            # nearest checkpoint so the search can recover an alternative branch.
            self._add_checkpoint(
                checkpoints, current_result, alternatives, initial_result
            )

            if self.enable_stagnation_expansion:
                if (
                    node_best_child_gain is not None
                    and node_best_child_gain > self.min_parent_gain
                ):
                    semantic_no_improve_steps = 0
                else:
                    semantic_no_improve_steps += 1

            if self.enable_checkpoint_backtracking:
                if best_result.score > previous_best_score:
                    backtrack_no_improve_steps = 0
                else:
                    backtrack_no_improve_steps += 1
                if (
                    self.backtrack_stagnation_patience is not None
                    and self.backtrack_stagnation_patience > 0
                    and backtrack_no_improve_steps
                    >= self.backtrack_stagnation_patience
                    and backtrack_uses < self.max_backtrack_uses
                ):
                    if self._activate_backtrack(
                        checkpoints, priority_queue, used_backtrack_texts
                    ):
                        backtrack_uses += 1
                        backtrack_no_improve_steps = 0
