"""MTEB-compatible encoder implementation wrapping sentence-transformers e5-base-v2.

Follows the official hackathon specification for PrePostPipelineEncoder(AbsEncoder).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

logger = logging.getLogger(__name__)


class PrePostPipelineEncoder(SentenceTransformerEncoderWrapper):
    """PrePostPipelineEncoder wrapping intfloat/e5-base-v2.

    Subclasses SentenceTransformerEncoderWrapper (which implements AbsEncoder) to
    remain 100% compliant with MTEB evaluation while providing clean query/document
    formatting and standalone encode methods.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name

        # Determine prompt format if not explicitly provided
        if model_prompts is not None:
            prompts = model_prompts
        elif "embeddinggemma" in model_name:
            prompts = {
                "query": "task: search result | query: ",
                "document": "title: none | text: ",
            }
        elif "e5" in model_name:
            prompts = {
                "query": "query: ",
                "document": "passage: ",
            }
        else:
            prompts = None

        # Load underlying SentenceTransformer instance with fallback for gated models
        if isinstance(model_name, str) and "embeddinggemma" in model_name:
            try:
                st_model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=kwargs.pop("trust_remote_code", False),
                )
            except Exception as e:
                logger.info("Primary load of %s failed (%s). Using RedHatAI mirror.", model_name, e)
                st_model = SentenceTransformer(
                    "RedHatAI/embeddinggemma-300m",
                    device=device,
                    trust_remote_code=kwargs.pop("trust_remote_code", False),
                )
            super().__init__(
                model=st_model,
                device=device,
                model_prompts=prompts,
                **kwargs,
            )
        else:
            super().__init__(
                model=model_name,
                device=device,
                model_prompts=prompts,
                **kwargs,
            )

        # Cap max_seq_length to 512 for fair CPU evaluation
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = 512

        # Ensure model meta is fully registered for MTEB task scoring
        try:
            self.mteb_model_meta = mteb.get_model_meta(model_name)
            if self.mteb_model_meta is not None:
                self.mteb_model_meta = self.mteb_model_meta.model_copy(
                    update={"name": model_name, "embed_dim": 768, "similarity_fn_name": "cosine"}
                )
        except Exception:
            if hasattr(self, "mteb_model_meta") and self.mteb_model_meta is not None:
                self.mteb_model_meta = self.mteb_model_meta.model_copy(
                    update={"name": model_name, "embed_dim": 768, "similarity_fn_name": "cosine"}
                )
            else:
                self.mteb_model_meta = ModelMeta.create_empty(
                    overwrites=dict(
                        name=model_name,
                        embed_dim=768,
                        similarity_fn_name="cosine",
                        loader=type(self),
                    )
                )

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode inputs for both the MTEB evaluation harness and standalone calls.

        Args:
            inputs: Either an MTEB DataLoader, or a Sequence[str].
            task_metadata: MTEB TaskMetadata (optional when called standalone).
            hf_split: Split name (default: 'test').
            hf_subset: Subset name (default: 'default').
            prompt_type: PromptType.query or PromptType.document.
            **kwargs: Additional encode kwargs like batch_size.

        Returns:
            Normalized 2D NumPy array of shape (N, embed_dim).
        """
        # Standalone usage with list/tuple of strings
        if isinstance(inputs, (list, tuple)) and all(isinstance(x, str) for x in inputs):
            prefix = ""
            if prompt_type is not None:
                pt_val = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)
                prefix = self.model_prompts.get(pt_val, "")

            texts = [f"{prefix}{t}" for t in inputs] if prefix else list(inputs)
            batch_size = kwargs.get("batch_size", 64)
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=kwargs.get("show_progress_bar", False),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        # MTEB harness execution
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )

    def encode_queries(self, queries: Sequence[str], **kwargs: Any) -> np.ndarray:
        """Encode query texts with the E5 query prefix and L2 normalization."""
        return self.encode(queries, prompt_type=PromptType.query, **kwargs)

    def encode_corpus(
        self, corpus: Sequence[str] | Sequence[dict[str, str]], **kwargs: Any
    ) -> np.ndarray:
        """Encode corpus code snippets with the E5 passage prefix and L2 normalization."""
        if isinstance(corpus, (list, tuple)) and corpus and isinstance(corpus[0], dict):
            texts = [doc.get("text", doc.get("content", "")) for doc in corpus]
        else:
            texts = list(corpus)  # type: ignore[arg-type]
        return self.encode(texts, prompt_type=PromptType.document, **kwargs)
