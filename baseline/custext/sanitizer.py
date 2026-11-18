"""CusText token sanitization.

This is a clean-room implementation of the mechanism in:

    S. Chen et al., "A Customized Text Sanitization Mechanism with
    Differential Privacy," Findings of ACL, 2023.

CusText first assigns input tokens to a small, shared output set of semantic
neighbours.  It then samples from that set with the exponential mechanism.
The shared support is the important part: for any two mapped tokens in one
group, each output can be emitted from both inputs.  ``conservative`` below
constructs disjoint output sets.  ``release_compatible`` additionally mirrors
the public repository's mutable-vector masking behavior, while ``aggressive``
is its overlapping-support ablation.

The public release maps words appearing in its training corpus and leaves
unknown words untouched.  That behaviour is faithfully available for the
word-vector mode, but does *not* protect tokens outside the mapping domain.
``from_model_token_embeddings`` is a practical comparison adapter that maps
every non-special token in the evaluated model's vocabulary instead.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


# This is the English list distributed with NLTK's ``stopwords`` corpus, which
# is what the authors' released CusText+ code uses.  Keeping it in-tree avoids
# a network download during a benchmark run.
NLTK_ENGLISH_STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you you're you've you'll you'd your yours
    yourself yourselves he him his himself she she's her hers herself it it's its
    itself they them their theirs themselves what which who whom this that that'll
    these those am is are was were be been being have has had having do does did
    doing a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down in
    out on off over under again further then once here there when where why how all
    any both each few more most other some such no nor not only own same so than too
    very s t can will just don don't should should've now d ll m o re ve y ain aren
    aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
    haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
    shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
    """.split()
)


@dataclass(frozen=True)
class CusTextConfig:
    """Parameters of a CusText sanitizer.

    ``epsilon`` is the per-token privacy budget used by the exponential
    mechanism.  As in the paper and authors' code, row-wise utilities are
    normalized to a range of length one, then sampled with
    ``softmax(epsilon * utility / 2)``.
    """

    epsilon: float
    top_k: int = 20
    metric: str = "cosine"  # ``cosine`` for Counter-Fitting, ``euclidean`` for GloVe.
    mapping_strategy: str = "release_compatible"
    tail_policy: str = "merge_last"
    seed: int = 42

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if self.top_k < 2:
            raise ValueError("top_k must be at least 2")
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if self.mapping_strategy not in {"conservative", "release_compatible", "aggressive"}:
            raise ValueError("mapping_strategy must be 'conservative', 'release_compatible', or 'aggressive'")
        if self.tail_policy not in {"merge_last", "allow_singleton"}:
            raise ValueError("tail_policy must be 'merge_last' or 'allow_singleton'")


@dataclass(frozen=True)
class _Distribution:
    outputs: Tuple[int, ...]
    probabilities: Tuple[float, ...]


def _runtime_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _normalise_rows(values: torch.Tensor) -> torch.Tensor:
    """Normalize each score row to [0, 1], handling degenerate rows safely."""

    lower = values.min(dim=1, keepdim=True).values
    upper = values.max(dim=1, keepdim=True).values
    span = upper - lower
    normalized = torch.where(span > 1e-12, (values - lower) / span, torch.zeros_like(values))
    return normalized


def _probabilities_for_group(
    representations: torch.Tensor,
    group: Sequence[int],
    config: CusTextConfig,
) -> torch.Tensor:
    """Return one exponential-mechanism row per input token in ``group``."""

    members = torch.as_tensor(group, dtype=torch.long, device=representations.device)
    vectors = representations.index_select(0, members)
    if config.metric == "cosine":
        # ``representations`` is unit-normalized in this branch.
        utility = vectors @ vectors.T
        utility = _normalise_rows(utility)
    else:
        # The paper uses negative normalized Euclidean distance as utility.
        distances = torch.cdist(vectors, vectors, p=2)
        utility = -_normalise_rows(distances)
    return torch.softmax(config.epsilon * utility / 2.0, dim=1)


def _choose_group(
    representations: torch.Tensor,
    squared_norms: torch.Tensor,
    available: torch.Tensor,
    seed_index: int,
    target_size: int,
    metric: str,
) -> List[int]:
    """Find semantic neighbours of one seed among currently available tokens."""

    query = representations[seed_index]
    if metric == "cosine":
        scores = representations @ query
        scores = scores.masked_fill(~available, -torch.inf)
        ranked = torch.topk(scores, k=target_size, largest=True).indices.tolist()
    else:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b; avoids a vocabulary-sized
        # [vocab, dim] temporary for every group.
        distances_sq = squared_norms + squared_norms[seed_index] - 2.0 * (representations @ query)
        distances_sq = distances_sq.masked_fill(~available, torch.inf)
        ranked = torch.topk(distances_sq, k=target_size, largest=False).indices.tolist()

    # A token must be in its own output set.  Self-similarity/self-distance
    # normally guarantees this, but explicit handling makes tied embeddings
    # deterministic and preserves CusText's K-to-K construction.
    if seed_index not in ranked:
        ranked[-1] = seed_index
    return [seed_index] + [idx for idx in ranked if idx != seed_index]


def _build_distributions(
    embeddings: np.ndarray,
    source_indices: Sequence[int],
    config: CusTextConfig,
    device: str,
    progress_desc: str,
    candidate_indices: Optional[Sequence[int]] = None,
) -> Dict[int, _Distribution]:
    """Build the mapping, returning an input-index to output distribution map."""

    config.validate()
    if embeddings.ndim != 2 or embeddings.shape[0] < config.top_k:
        raise ValueError("embedding table must be two-dimensional and contain at least top_k entries")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding table contains non-finite values")

    n_tokens = int(embeddings.shape[0])
    unique_sources = list(dict.fromkeys(int(index) for index in source_indices))
    if not unique_sources:
        raise ValueError("source_indices is empty")
    if min(unique_sources) < 0 or max(unique_sources) >= n_tokens:
        raise ValueError("source_indices contains an out-of-range token index")
    if candidate_indices is None:
        unique_candidates = list(range(n_tokens))
    else:
        unique_candidates = list(dict.fromkeys(int(index) for index in candidate_indices))
        if not unique_candidates:
            raise ValueError("candidate_indices is empty")
        if min(unique_candidates) < 0 or max(unique_candidates) >= n_tokens:
            raise ValueError("candidate_indices contains an out-of-range token index")
    candidate_set = set(unique_candidates)
    if not set(unique_sources).issubset(candidate_set):
        raise ValueError("every source token must be an eligible output candidate")
    if len(unique_candidates) < config.top_k:
        raise ValueError("fewer than top_k output candidates are available")

    torch_device = _runtime_device(device)
    representations = torch.as_tensor(embeddings, dtype=torch.float32, device=torch_device)
    if config.metric == "cosine":
        representations = torch.nn.functional.normalize(representations, p=2, dim=1, eps=1e-12)
    squared_norms = (representations * representations).sum(dim=1)

    groups: List[List[int]] = []
    if config.mapping_strategy == "aggressive":
        available = torch.zeros(n_tokens, dtype=torch.bool, device=torch_device)
        available[torch.as_tensor(unique_candidates, dtype=torch.long, device=torch_device)] = True
        for seed_index in tqdm(unique_sources, desc=progress_desc):
            groups.append(
                _choose_group(
                    representations,
                    squared_norms,
                    available,
                    seed_index,
                    config.top_k,
                    config.metric,
                )
            )
        distributions: Dict[int, _Distribution] = {}
        for seed_index, group in zip(unique_sources, groups):
            probabilities = _probabilities_for_group(representations, group, config)[0].detach().cpu().tolist()
            distributions[seed_index] = _Distribution(tuple(group), tuple(float(item) for item in probabilities))
        return distributions

    if config.mapping_strategy == "release_compatible":
        # The authors' release applies the default ``conservative`` label to a
        # mutable-embedding procedure: after assigning a group it overwrites
        # its candidate vectors with zero (cosine) or a large value (Euclidean)
        # before finding the next group.  It is not equivalent to a strict
        # Boolean candidate mask when a zeroed vector can reappear among a
        # later seed's top-K neighbours.  Preserve that behaviour under an
        # explicit name so paper-result matching and formal support claims are
        # never silently conflated.
        mutable = representations.clone()
        mutable_norms = (mutable * mutable).sum(dim=1)
        all_candidates = torch.zeros(n_tokens, dtype=torch.bool, device=torch_device)
        all_candidates[torch.as_tensor(unique_candidates, dtype=torch.long, device=torch_device)] = True
        distributions: Dict[int, _Distribution] = {}
        assigned = set()
        for seed_index in tqdm(unique_sources, desc=progress_desc):
            if seed_index in assigned:
                continue
            group = _choose_group(
                mutable,
                mutable_norms,
                all_candidates,
                seed_index,
                config.top_k,
                config.metric,
            )
            group = list(dict.fromkeys(group))
            probabilities = _probabilities_for_group(mutable, group, config).detach().cpu().numpy()
            outputs = tuple(int(item) for item in group)
            for row, token_index in enumerate(group):
                if token_index not in assigned:
                    distributions[int(token_index)] = _Distribution(
                        outputs,
                        tuple(float(item) for item in probabilities[row].tolist()),
                    )
                    assigned.add(int(token_index))
            group_tensor = torch.as_tensor(group, dtype=torch.long, device=torch_device)
            if config.metric == "cosine":
                mutable[group_tensor] = 0.0
            else:
                mutable[group_tensor] = 1.0e9
            mutable_norms = (mutable * mutable).sum(dim=1)
        return distributions

    # Conservative mapping: an output candidate is consumed once, yielding
    # shared, non-overlapping output supports.  This is the strategy used by
    # the public release's default CLI arguments.
    available = torch.zeros(n_tokens, dtype=torch.bool, device=torch_device)
    available[torch.as_tensor(unique_candidates, dtype=torch.long, device=torch_device)] = True
    remaining = len(unique_candidates)
    assigned = set()
    for seed_index in tqdm(unique_sources, desc=progress_desc):
        if seed_index in assigned:
            continue
        group_size = min(config.top_k, remaining)
        group = _choose_group(
            representations,
            squared_norms,
            available,
            seed_index,
            group_size,
            config.metric,
        )
        # The de-duplication is relevant only when there are exactly tied
        # neighbours and the forced seed replacement above was required.
        group = list(dict.fromkeys(group))
        groups.append(group)
        available[torch.as_tensor(group, dtype=torch.long, device=torch_device)] = False
        remaining -= len(group)
        assigned.update(group)

    # When mapping a complete finite vocabulary the final group can contain a
    # single token.  Merge it with the preceding group so every mapped input
    # has at least two possible outputs.  This is a safe strengthening of the
    # paper pseudocode's short final group; ``allow_singleton`` is available
    # for exact ablations of that edge case.
    if config.tail_policy == "merge_last" and len(groups) >= 2 and len(groups[-1]) == 1:
        groups[-2].extend(groups[-1])
        groups.pop()

    distributions = {}
    for group in groups:
        probabilities = _probabilities_for_group(representations, group, config).detach().cpu().numpy()
        outputs = tuple(int(item) for item in group)
        for row, token_index in enumerate(group):
            distributions[int(token_index)] = _Distribution(
                outputs,
                tuple(float(item) for item in probabilities[row].tolist()),
            )
    return distributions


def load_text_embeddings(path: str | Path, max_vectors: int = 0) -> Tuple[List[str], np.ndarray]:
    """Load a whitespace-delimited word-vector file.

    Both standard ``word value1 ...`` files and files with a leading
    ``<vocab_size> <dimension>`` header are accepted.  ``max_vectors`` is for
    smoke tests only; use zero to load the complete representation space.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Word-vector file not found: {path}")

    tokens: List[str] = []
    rows: List[np.ndarray] = []
    dimension: Optional[int] = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip().split()
            if not fields:
                continue
            if line_number == 1 and len(fields) == 2:
                try:
                    int(fields[0])
                    dimension = int(fields[1])
                    continue
                except ValueError:
                    pass
            if len(fields) < 2:
                continue
            token, values = fields[0], fields[1:]
            try:
                vector = np.asarray(values, dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric vector at {path}:{line_number}") from exc
            if dimension is None:
                dimension = int(vector.size)
            if vector.size != dimension:
                raise ValueError(
                    f"Inconsistent vector dimension at {path}:{line_number}; "
                    f"expected {dimension}, got {vector.size}"
                )
            tokens.append(token)
            rows.append(vector)
            if max_vectors > 0 and len(rows) >= max_vectors:
                break
    if not rows:
        raise ValueError(f"No usable word vectors found in {path}")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"Word-vector file contains duplicate tokens: {path}")
    return tokens, np.stack(rows, axis=0)


class CusTextSanitizer:
    """A fitted CusText mapping for words or model token IDs."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        kind: str,
        config: CusTextConfig,
        vocabulary: Optional[Sequence[str]],
        distributions: Mapping[int, _Distribution],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if kind not in {"word", "model_token"}:
            raise ValueError("kind must be 'word' or 'model_token'")
        config.validate()
        self.kind = kind
        self.config = config
        self.vocabulary = list(vocabulary) if vocabulary is not None else None
        self.distributions = dict(distributions)
        self.metadata = dict(metadata or {})
        self._token_to_index = (
            {token: index for index, token in enumerate(self.vocabulary)} if self.vocabulary is not None else None
        )
        if self.kind == "word" and self.vocabulary is None:
            raise ValueError("word sanitizers require a vocabulary")

    @classmethod
    def from_word_vectors(
        cls,
        vocabulary: Sequence[str],
        embeddings: np.ndarray,
        source_words: Iterable[str],
        config: CusTextConfig,
        *,
        device: str = "cuda:0",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "CusTextSanitizer":
        """Fit the word-level protocol used in the original CusText release.

        Only source words observed in the selected mapping corpus are mapped.
        This deliberately matches the public code and lets the runner report
        its coverage on a held-out split.
        """

        if len(vocabulary) != int(embeddings.shape[0]):
            raise ValueError("vocabulary and embeddings have different lengths")
        token_to_index = {token: index for index, token in enumerate(vocabulary)}
        if len(token_to_index) != len(vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")
        counts = Counter(word for word in source_words if word in token_to_index)
        # Counter.most_common reproduces the public implementation's frequency
        # ordering, including first-occurrence tie breaking on modern Python.
        source_indices = [token_to_index[word] for word, _ in counts.most_common()]
        distributions = _build_distributions(
            embeddings,
            source_indices,
            config,
            device,
            progress_desc="Building CusText word mapping",
        )
        merged_metadata: Dict[str, object] = {
            "source_domain": "observed_words",
            "source_words_with_vectors": len(source_indices),
            "embedding_vocab_size": len(vocabulary),
        }
        merged_metadata.update(metadata or {})
        return cls(
            kind="word",
            config=config,
            vocabulary=vocabulary,
            distributions=distributions,
            metadata=merged_metadata,
        )

    @classmethod
    def from_model_token_embeddings(
        cls,
        embeddings: torch.Tensor | np.ndarray,
        special_token_ids: Iterable[int],
        config: CusTextConfig,
        *,
        device: str = "cuda:0",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "CusTextSanitizer":
        """Fit the complete-vocabulary model-token adaptation.

        It is intentionally labelled an adaptation: CusText was proposed over
        word embeddings and reconstructed text, while this variant substitutes
        non-special token IDs before the same model consumes them.  Its benefit
        for DP-SPD is that all non-special model tokens are in the domain.
        """

        array = embeddings.detach().float().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        if array.ndim != 2:
            raise ValueError("model input embeddings must be a two-dimensional tensor")
        special = {int(token_id) for token_id in special_token_ids}
        source_indices = [index for index in range(int(array.shape[0])) if index not in special]
        if len(source_indices) < config.top_k:
            raise ValueError("fewer than top_k non-special tokens are available")
        distributions = _build_distributions(
            np.asarray(array, dtype=np.float32),
            source_indices,
            config,
            device,
            progress_desc="Building CusText model-token mapping",
            candidate_indices=source_indices,
        )
        merged_metadata: Dict[str, object] = {
            "source_domain": "all_non_special_model_tokens",
            "model_vocab_size": int(array.shape[0]),
            "mapped_token_count": len(distributions),
            "special_token_ids": sorted(special),
        }
        merged_metadata.update(metadata or {})
        return cls(kind="model_token", config=config, vocabulary=None, distributions=distributions, metadata=merged_metadata)

    @property
    def mapped_input_count(self) -> int:
        return len(self.distributions)

    def group_size_range(self) -> Tuple[int, int]:
        if not self.distributions:
            return (0, 0)
        sizes = [len(item.outputs) for item in self.distributions.values()]
        return min(sizes), max(sizes)

    @staticmethod
    def _probability_array(distribution: _Distribution) -> np.ndarray:
        # Mapping probabilities originate from float32 Torch softmax values.
        # NumPy's Generator.choice validates the sum more strictly after the
        # conversion to float64, so normalize once at the sampling boundary.
        values = np.asarray(distribution.probabilities, dtype=np.float64)
        total = float(values.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("CusText mapping has an invalid probability row")
        return values / total

    def _sample(self, distribution: _Distribution, rng: np.random.Generator) -> int:
        return int(rng.choice(np.asarray(distribution.outputs), p=self._probability_array(distribution)))

    def sanitize_words(
        self,
        words: Sequence[str],
        *,
        rng: np.random.Generator,
        preserve_stopwords: bool = False,
        numeric_policy: str = "release_compatible",
    ) -> Tuple[List[str], Dict[str, int]]:
        """Sanitize whitespace tokens and return output plus coverage counters."""

        if self.kind != "word" or self._token_to_index is None or self.vocabulary is None:
            raise RuntimeError("sanitize_words is available only for a word-vector sanitizer")
        if numeric_policy not in {"release_compatible", "keep"}:
            raise ValueError("numeric_policy must be 'release_compatible' or 'keep'")
        output: List[str] = []
        counters: MutableMapping[str, int] = Counter()
        for word in words:
            counters["total"] += 1
            if preserve_stopwords and word in NLTK_ENGLISH_STOPWORDS:
                counters["stopword_preserved"] += 1
                output.append(word)
                continue
            token_index = self._token_to_index.get(word)
            distribution = self.distributions.get(token_index) if token_index is not None else None
            if distribution is not None:
                counters["mapped"] += 1
                output.append(self.vocabulary[self._sample(distribution, rng)])
                continue
            counters["unmapped"] += 1
            if numeric_policy == "release_compatible":
                try:
                    output.append(str(round(float(word)) + int(rng.integers(0, 1000))))
                    counters["numeric_perturbed"] += 1
                    continue
                except ValueError:
                    pass
            output.append(word)
        return output, dict(counters)

    def sanitize_text(
        self,
        text: str,
        *,
        rng: np.random.Generator,
        preserve_stopwords: bool = False,
        numeric_policy: str = "release_compatible",
    ) -> Tuple[str, Dict[str, int]]:
        words, counters = self.sanitize_words(
            str(text).split(),
            rng=rng,
            preserve_stopwords=preserve_stopwords,
            numeric_policy=numeric_policy,
        )
        return " ".join(words), counters

    def sanitize_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Apply the model-token adaptation to a token-ID batch or tensor."""

        if self.kind != "model_token":
            raise RuntimeError("sanitize_token_ids is available only for a model-token sanitizer")
        ids = input_ids.detach().cpu().numpy()
        result = ids.copy()
        active = np.ones_like(ids, dtype=bool) if attention_mask is None else attention_mask.detach().cpu().numpy().astype(bool)
        counters: MutableMapping[str, int] = Counter(total=int(active.sum()))
        for token_id in np.unique(ids[active]):
            distribution = self.distributions.get(int(token_id))
            positions = active & (ids == token_id)
            count = int(positions.sum())
            if distribution is None:
                counters["unmapped"] += count
                continue
            counters["mapped"] += count
            result[positions] = rng.choice(
                np.asarray(distribution.outputs, dtype=np.int64),
                size=count,
                p=self._probability_array(distribution),
            )
        return torch.as_tensor(result, dtype=input_ids.dtype), dict(counters)

    def save(self, path: str | Path) -> None:
        """Persist a fitted mapping without storing the potentially large vectors."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "kind": self.kind,
            "config": asdict(self.config),
            "vocabulary": self.vocabulary,
            "metadata": self.metadata,
            "distributions": {
                str(index): {"outputs": list(item.outputs), "probabilities": list(item.probabilities)}
                for index, item in self.distributions.items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> "CusTextSanitizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported CusText mapping schema in {path}")
        distributions = {
            int(index): _Distribution(
                tuple(int(item) for item in value["outputs"]),
                tuple(float(item) for item in value["probabilities"]),
            )
            for index, value in payload["distributions"].items()
        }
        return cls(
            kind=payload["kind"],
            config=CusTextConfig(**payload["config"]),
            vocabulary=payload.get("vocabulary"),
            distributions=distributions,
            metadata=payload.get("metadata"),
        )
