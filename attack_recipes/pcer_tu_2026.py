# ! /usr/bin/nev python
# -*-coding:utf8-*-
# ! /usr/bin/nev python
# -*- coding: utf-8 -*-

"""Controlled fusion attack recipe for experiment 46."""

from textattack import Attack
from textattack.constraints.pre_transformation import RepeatModification
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BFSTestHouDMod46
from textattack.search_methods.bfs_test_houd2_46 import WordSwapWordNetEmbeddingSwitch

from .attack_recipe import AttackRecipe


class BFSTuPriorWIRHouDMod46(AttackRecipe):
    """Controlled fusion recipe for experiment 46."""

    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapWordNetEmbeddingSwitch(
            embedding_max_candidates=50,
        )

        constraints = [
            RepeatModification(),
        ]

        search_method = BFSTestHouDMod46(
            wir_method="delete",
            append_low_priority=False,
            prior_score_weight=0.05,
            candidate_score_weight=1.0,
            candidate_probe_top_k=3,
            first_round_probe_all=True,
            full_probe_every_n_modifications=2,
            base_wir_top_k=5,
            high_prior_reprobe_interval=2,
            exploration_rate=0.05,
            flip_bonus=1000.0,
            min_word_len=3,
            random_seed=None,
            min_parent_gain=0.0,
            skip_modified_indices=True,
            embedding_promotion_top_k=8,
            embedding_promotion_rank_limit=5,
            stagnation_k1=2,
            stagnation_k2=4,
            embedding_top_k1=5,
            embedding_top_k2=15,
            rank_limit_k1=3,
            rank_limit_k2=5,
            max_restore_attempts=None,
            max_refine_positions=None,
            max_refine_candidates=3,
            require_lexical_improvement=True,
            max_backtrack_uses=2,
            alternative_pool_size=32,
            local_gain_floor=0.0,
            backtrack_stagnation_patience=3,
        )

        goal_function = UntargetedClassification(model_wrapper)

        return Attack(
            goal_function,
            constraints,
            transformation,
            search_method,
        )
