from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from qa_service.exceptions import MultiplePoolingMethodsException


class EmbeddingLayer:
    def __init__(self, model_name: str, use_gpu: bool = False):
        self.model_name: str = model_name

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model: AutoModel = AutoModel.from_pretrained(self.model_name)
        self.use_gpu: bool = False
        if use_gpu:
            if torch.cuda.is_available():
                self.use_gpu = True
            else:
                print(
                    "Attempted to use GPU but pytorch is not set up correctly, continuing on CPU."
                )

        if self.use_gpu:
            self.model.cuda()

        self.model.eval()

    def tokenize_text(
        self, text_to_tokenize: Union[List[str], str], padding: Union[bool, str] = 'max_length'
    ) -> BatchEncoding:
        """
        Given a string or list of strings,
        this method will encode and tokenize
        each string. Returns the tokenized
        texts as a BatchEncoding.

        Arguments:
            text_to_tokenize : Union[List[str], str]
                Either a string or list of strings
                to be tokenized.
            padding : Union[bool, str]
                Either True/False, or a padding strategy
                to apply to the batch of `text_to_tokenize`.
        Returns:
            tokenized_text : BatchEncoding
                A dictionary-like object containing
                the tokenized text and attention mask
                from any padding and/or truncating
                performed.
        """

        if isinstance(text_to_tokenize, str):
            text_to_tokenize: list = [text_to_tokenize]

        tokenized_text: BatchEncoding = self.tokenizer(
            text_to_tokenize,
            return_tensors="pt",
            padding=padding,
            truncation=True,
        )
        return tokenized_text

    def generate_embeddings(self, tokens_to_embed: BatchEncoding) -> BaseModelOutput:
        """
        Given a pytorch Tensor containing the tokenized
        text you wish to embed, this method embeds the
        tokens into an embedding matrix and returns
        the embeddings as a pytorch Tensor.

        Arguments:
            tokens_to_embed : BaseModelOutput
                 pytorch Tensor containing tokenized text.

        Returns:

        """

        if self.use_gpu:
            tokens_to_embed.cuda()

        embeddings: BaseModelOutput = self.model(**tokens_to_embed)
        return embeddings

    def pool_embeddings(
        self,
        embeddings_to_pool: torch.Tensor,
        mean_pooling: bool = True,
        return_cls_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Given the generated embeddings as a pytorch Tensor,
        this method performs pooling on those embeddings
        to provide a [samples, hidden_dimension] vector.

        Arguments:
            embeddings_to_pool : torch.Tensor
                The embeddings or last_hidden_state
                you wish to pool.
            mean_pooling : bool
                True if you wish to apply mean pooling.
            return_cls_embedding : bool
                True if you wish to return the last CLS
                vector.

        Returns:
            pooled_vector : torch.Tensor
                Pooled `embeddings_to_pool` vector.
        """

        if mean_pooling and return_cls_embedding:
            raise MultiplePoolingMethodsException

        pooled_vector: torch.Tensor = torch.tensor([])
        if mean_pooling:
            pooled_vector: torch.Tensor = embeddings_to_pool.mean(1)

        elif return_cls_embedding:
            pooled_vector: torch.Tensor = embeddings_to_pool[:, 0, :]

        return pooled_vector

    def __call__(
        self,
        text_to_tokenize: Union[List[str], str],
        padding: Union[bool, str] = 'max_length',
        mean_pooling: bool = True,
        return_cls_embedding: bool = False,
    ) -> torch.Tensor:
        """
        All-in-one convenience method for handling
        all tokenization, embedding, and pooling.

        Arguments:
            text_to_tokenize : Union[List[str], str]
                Either a string or list of strings
                to be tokenized.
            padding : Union[bool, str]
                Either True/False, or a padding strategy
                to apply to the batch of `text_to_tokenize`.
            mean_pooling : bool
                True if you wish to apply mean pooling.
            return_cls_embedding : bool
                True if you wish to return the last CLS

        Returns:
            pooled_vector : torch.Tensor
                Pooled `embeddings_to_pool` vector.
        """

        if mean_pooling and return_cls_embedding:
            raise MultiplePoolingMethodsException

        if isinstance(text_to_tokenize, str):
            text_to_tokenize = [text_to_tokenize]

        out: BatchEncoding = self.tokenize_text(text_to_tokenize, padding)
        out: BaseModelOutput = self.generate_embeddings(out)
        out: torch.Tensor = out.last_hidden_state
        out: torch.Tensor = self.pool_embeddings(
            out, mean_pooling=mean_pooling, return_cls_embedding=return_cls_embedding
        )
        return out
