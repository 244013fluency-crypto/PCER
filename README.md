# PCER

Code release for the paper:

> **PCER: An Effective and Efficient Method for Textual Adversarial Attacks**  
> Jiahe Tu, Mingxuan Xiao, Yan Xiao, Shunhui Ji, and Pengcheng Zhang

PCER is a word-level black-box textual adversarial attack method designed to improve both **attack effectiveness** and **attack efficiency**. It combines a WIR-guided best-first search with four complementary mechanisms for position prioritization, candidate-space exploration, search-path recovery, and post-success refinement.

The name **PCER** reflects three core operations in the attack process—**P**rior filtering, dynamic **C**andidate awareness, candidate-neighborhood **E**xpansion, and adversarial-example **R**ecovery—together with nearest-checkpoint path backtracking.

---

## An Example of Textual Adversarial Vulnerability

Language models can be sensitive to subtle word-level perturbations even when the overall semantics of the input are largely preserved.

The following example illustrates a military intelligence scenario in which a frontline intelligence officer sends a report to a command center. A subtle perturbation to a decision-critical word causes the target model to misclassify the received intelligence, which may subsequently affect decision support.

<p align="center">
  <img src="images/figure1.png" width="760">
</p>


<p align="center">
  <b>Figure 1.</b> Adversarial example in a military intelligence scenario.
</p>


---

## Overview of PCER

Given an original text \(x\), a task-oriented prompt \(P\), and a victim model \(F\), PCER forms the complete input \(I_0=[P;x]\) and performs a word-level untargeted adversarial attack. The objective is to find an adversarial input \(I_{\mathrm{adv}}\) that changes the original model prediction while satisfying predefined semantic, perturbation, and query-budget constraints.

<p align="center">
  <img src="images/figure2.png" width="860">
</p>


<p align="center">
  <b>Figure 2.</b> Overall workflow of the PCER method.
</p>


PCER first constructs synonym candidates, defines the attack objective and constraints, and initializes a priority queue for best-first search. During iterative search, it filters low-value positions, computes word-importance and candidate-awareness scores, ranks perturbation positions, performs synonym substitutions, and updates the priority queue and the globally best state. When search becomes unproductive, PCER can expand the candidate neighborhood or revisit a promising alternative path. After attack success, it further refines the adversarial example by removing redundant perturbations while preserving attack success.

---

## Core Components

PCER contains four complementary components.

### 1. Prior Filtering and Dynamic Candidate Awareness

Conventional WIR-based attacks primarily rank positions according to word importance. However, an influential word may contain few useful substitution candidates.

PCER first filters stopwords, low-value function words, and invalid positions while assigning higher priority to task-relevant words. For each retained position, PCER jointly considers:

- the Word Importance Ranking (WIR) score;
- prior information; and
- candidate utility estimated from a small number of substitution candidates.

Dynamic candidate awareness reduces repeated probing by reusing previously evaluated candidates and restricting candidate evaluation to promising positions when full probing is unnecessary.

### 2. Stagnation-Triggered Candidate Neighborhood Expansion

PCER uses WordNet as the default candidate space. A position is treated as locally stagnant when it has no valid candidate or when none of its candidates improves the current attack state.

Instead of using a uniformly enlarged candidate space throughout the attack, PCER expands the candidate neighborhood only when the default neighborhood becomes unproductive. This provides additional substitution opportunities while limiting unnecessary candidate evaluations during normal search.

### 3. Nearest-Checkpoint Path Backtracking

Best-first search favors globally promising states, but locally improving alternatives may be discarded when they do not exceed the current global best.

PCER stores such locally improving states as backup branches at selected checkpoints. When the global best remains unchanged for several expansions or the priority queue becomes empty, the search resumes from the nearest available checkpoint. The numbers of retained checkpoints and backtracking operations are bounded to avoid excessive exploration.

### 4. Success-Preserving Adversarial Example Recovery

A successful adversarial example may still contain substitutions that are unnecessary for maintaining attack success.

PCER revisits the modified positions after a successful attack. It first attempts to restore each perturbed word to its original form. If direct restoration fails, PCER considers synonyms that are lexically closer to the original word. A restoration or replacement is retained only when the adversarial effect is preserved.

This post-processing procedure reduces redundant perturbations and improves the naturalness of successful adversarial examples.

---

## Component Configuration

To make the contribution of each component easy to inspect and reproduce, the PCER-specific mechanisms in `pcer.py` are independently configurable and **disabled by default**.

With all switches set to `False`, the implementation behaves as a basic WIR-guided best-first search using the default WordNet candidate space.

```python
from pcer import PCERSearch

search_method = PCERSearch(
    enable_prior_candidate_aware=False,
    enable_stagnation_expansion=False,
    enable_checkpoint_backtracking=False,
    enable_success_restoration=False,
)
```

The four switches correspond to:

| Switch                           | Component                                             | Main purpose                                                 |
| -------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| `enable_prior_candidate_aware`   | Prior Filtering + Dynamic Candidate Awareness         | Reduce low-value position evaluation and candidate queries   |
| `enable_stagnation_expansion`    | Stagnation-Triggered Candidate Neighborhood Expansion | Improve local exploration when the default candidate neighborhood is ineffective |
| `enable_checkpoint_backtracking` | Nearest-Checkpoint Path Backtracking                  | Revisit promising alternative search paths                   |
| `enable_success_restoration`     | Success-Preserving Adversarial Example Recovery       | Remove redundant perturbations while preserving attack success |

To run the complete PCER method:

```python
from pcer import PCERSearch

search_method = PCERSearch(
    enable_prior_candidate_aware=True,
    enable_stagnation_expansion=True,
    enable_checkpoint_backtracking=True,
    enable_success_restoration=True,
)
```

The switches can also be enabled individually or incrementally to examine the effect of each component.

---

## Datasets

We evaluate PCER on three widely used sentiment-classification datasets:

- **SST-2**
- **IMDB**
- **MR**

SST-2 and MR mainly contain short movie reviews, whereas IMDB contains relatively long movie reviews.

---

## Victim Models

The experiments consider four victim models:

- **BERT-base-uncased**
- **RoBERTa**
- **Mistral-7B**
- **Phi-4**

This setting covers both conventional pretrained language models and large language models.

---

## Baselines

PCER is compared with five representative textual adversarial attack methods:

- **ABS**
- **ABFS**
- **BinarySelect**
- **TextCheater**
- **TextFooler**

These methods represent different search strategies, including adaptive search, best-first search, binary-search-based position selection, Tabu search, and WIR-guided synonym substitution.

---

## Evaluation Metrics

We evaluate attack effectiveness, adversarial-example quality, and attack efficiency using six metrics:

| Metric     | Description                                                  | Preferred |
| ---------- | ------------------------------------------------------------ | --------- |
| **ASR**    | Attack Success Rate                                          | Higher    |
| **C-rate** | Percentage of modified words in successful adversarial examples | Lower     |
| **PPL**    | Perplexity of generated adversarial examples                 | Lower     |
| **G-E**    | Number of grammatical errors                                 | Lower     |
| **Q-N**    | Average victim-model queries per successful adversarial example | Lower     |
| **T-O**    | Average generation time per successful adversarial example   | Lower     |

Grammatical errors are measured using LanguageTool.

---

## Main Results

The experiments in the current manuscript show that:

- PCER achieves the highest attack success rate in all **12 model-dataset settings**, with an average ASR of **88.73%**.
- On **Phi-4 with IMDB**, PCER achieves an ASR of **98.25%**, compared with **38.50%** for BinarySelect.
- PCER consistently requires fewer victim-model queries and less generation time than the strong efficiency baselines considered in the paper.
- PCER achieves the highest transfer ASR in all **12 reported cross-model transfer settings**, with an average transfer ASR of **34.78%**.
- Ablation experiments indicate complementary roles for the four components: prior filtering and candidate awareness mainly improve efficiency; candidate-neighborhood expansion and path backtracking mainly improve attack success; adversarial-example recovery mainly improves naturalness.

---

## Repository Structure

PCER is implemented on top of the **TextAttack** framework and follows its modular organization. The main modules relevant to adversarial-example generation include:

```text
PCER/
├── attack_recipes/       # Attack configurations
├── constraints/          # Linguistic and semantic constraints
├── datasets/             # Dataset interfaces
├── goal_functions/       # Attack objectives and success criteria
├── models/               # Victim-model wrappers
├── search_methods/       # Search algorithms
├── transformations/      # Text transformations and word substitutions
├── pcer.py               # Core PCER search implementation
├── LICENSE
└── README.md
```

The most important file for the proposed method is:

- **`pcer.py`**: implements the WIR-guided best-first search together with the four configurable PCER components.

---

## Dependencies

PCER is developed in a TextAttack-based environment. A compatible environment is:

```text
bert-score>=0.3.5
autocorrect==2.6.1
accelerate==0.25.0
datasets==2.15.0
nltk==3.8.1
openai==1.3.7
sentencepiece==0.1.99
tokenizers==0.15.0
torch==2.1.1
tqdm==4.66.1
transformers==4.38.0
Pillow==10.3.0
transformers_stream_generator==0.0.5
matplotlib==3.8.3
tiktoken==0.6.0
```

The exact dependencies required for a specific experiment may vary with the selected victim model.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/244013fluency-crypto/PCER.git
cd PCER
```

Install the local package in editable mode:

```bash
pip install -e . ".[dev]"
```

---

## How to Run

The core search method can be imported directly from `pcer.py`.

For the complete PCER configuration:

```python
from pcer import PCERSearch

search_method = PCERSearch(
    enable_prior_candidate_aware=True,
    enable_stagnation_expansion=True,
    enable_checkpoint_backtracking=True,
    enable_success_restoration=True,
)
```

A complete attack experiment should additionally specify:

- the dataset;
- the victim model;
- the transformation method;
- linguistic and semantic constraints;
- the goal function; and
- the query budget.

After the final experiment entry script is selected, the corresponding command can be added here, for example:

```bash
python <experiment_script>.py
```

---

## Reproducing Component Effects

The configurable implementation allows users to study the contribution of each mechanism without modifying the search code.

```python
# Basic WIR-guided best-first search
PCERSearch()

# + Prior filtering and dynamic candidate awareness
PCERSearch(
    enable_prior_candidate_aware=True,
)

# + Stagnation-triggered candidate neighborhood expansion
PCERSearch(
    enable_prior_candidate_aware=True,
    enable_stagnation_expansion=True,
)

# + Nearest-checkpoint path backtracking
PCERSearch(
    enable_prior_candidate_aware=True,
    enable_stagnation_expansion=True,
    enable_checkpoint_backtracking=True,
)

# Full PCER
PCERSearch(
    enable_prior_candidate_aware=True,
    enable_stagnation_expansion=True,
    enable_checkpoint_backtracking=True,
    enable_success_restoration=True,
)
```

This interface is intended to make the behavior of each component transparent and facilitate reproducibility and ablation analysis.

---

## License

Please refer to the [LICENSE](LICENSE) file for the terms governing the use of this repository.

---

## Acknowledgement

This project is developed based on the [TextAttack](https://github.com/QData/TextAttack) framework. We sincerely thank the TextAttack authors and contributors for providing an extensible platform for adversarial attacks in NLP.

---

## Citation

If you find this repository useful in your research, please consider citing our paper:

> **PCER: An Effective and Efficient Method for Textual Adversarial Attacks**  
> Jiahe Tu, Mingxuan Xiao, Yan Xiao, Shunhui Ji, and Pengcheng Zhang.

Formal BibTeX information will be added after the publication information becomes available.

---

## Contact

For questions regarding the code or experiments, please open an issue in this repository.
