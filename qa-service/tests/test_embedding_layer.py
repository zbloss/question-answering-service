import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from qa_service.embedding_layer import EmbeddingLayer
from qa_service.exceptions import MultiplePoolingMethodsException


class TestEmbeddingLayer:
    model_name: str = "distilbert-base-uncased"
    example_text: list = ["Hello there.", "Where are you?", "I'm so sorry"]

    embedding_layer: EmbeddingLayer = EmbeddingLayer(
        model_name=model_name, use_gpu=False
    )

    embedding_size: int = embedding_layer.model.config.max_position_embeddings
    hidden_dimension: int = embedding_layer.model.config.dim

    def test_init(self):
        assert isinstance(
            self.embedding_layer.model_name, str
        ), f"self.embedding_layer.model_name ({type(self.embedding_layer.model_name)}) is not a string as expected."
        assert isinstance(
            self.embedding_layer, EmbeddingLayer
        ), f"self.embedding_layer ({type(self.embedding_layer)}) is not an EmbeddingLayer as expected."
        assert (
            self.embedding_layer.use_gpu == False
        ), f"self.embedding_layer.use_gpu ({self.embedding_layer.use_gpu}) is not False as expected."

    def test_model_eval_model(self):
        assert (
            self.embedding_layer.model.training == False
        ), f"Model is not in evaluation model as expected."

    def test_tokenize_single_text(self):
        single_text: str = self.example_text[0]
        assert isinstance(single_text, str)

        tokens: BatchEncoding = self.embedding_layer.tokenize_text(single_text)
        assert isinstance(
            tokens, BatchEncoding
        ), f"tokens ({type(tokens)}) were not properly returned as BatchEncoding."

        assert hasattr(tokens, "input_ids"), f"tokens.input_ids does not exist."
        assert hasattr(tokens, "attention_mask"), f"tokens.input_ids does not exist."

    def test_tokenize_multiple_texts(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)
        assert isinstance(
            tokens, BatchEncoding
        ), f"tokens ({type(tokens)}) were not properly returned as BatchEncoding."

        assert hasattr(tokens, "input_ids"), f"tokens.input_ids does not exist."
        assert hasattr(tokens, "attention_mask"), f"tokens.input_ids does not exist."

    def test_generate_embeddings(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        assert isinstance(
            embeddings, BaseModelOutput
        ), f"embeddings ({type(embeddings)}) is not a BaseModelOutput as expected."

        assert hasattr(
            embeddings, "last_hidden_state"
        ), f"embeddings.last_hidden_state does not exist."

        expected_shape: torch.Size = torch.Size(
            [len(self.example_text), self.embedding_size, self.hidden_dimension]
        )

        assert (
            embeddings.last_hidden_state.shape == expected_shape
        ), f"embeddings.last_hidden_state.shape ({embeddings.last_hidden_state.shape}) is an invalid shape"

    def test_multiple_pooling_methods_exception(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        last_hidden_state: torch.Tensor = embeddings.last_hidden_state

        with pytest.raises(MultiplePoolingMethodsException):
            self.embedding_layer.pool_embeddings(
                last_hidden_state, mean_pooling=True, return_cls_embedding=True
            )

    def test_mean_pooling(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        last_hidden_state: torch.Tensor = embeddings.last_hidden_state

        pooled_output: torch.Tensor = self.embedding_layer.pool_embeddings(
            last_hidden_state, mean_pooling=True, return_cls_embedding=False
        )

        expected_shape: torch.Size = torch.Size(
            [len(self.example_text), self.hidden_dimension]
        )
        assert (
            pooled_output.shape == expected_shape
        ), f"Unexpected shape ({pooled_output.shape}) from mean_pooling=True"

    def test_return_cls_embedding(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        last_hidden_state: torch.Tensor = embeddings.last_hidden_state

        pooled_output: torch.Tensor = self.embedding_layer.pool_embeddings(
            last_hidden_state, mean_pooling=False, return_cls_embedding=True
        )

        expected_shape: torch.Size = torch.Size(
            [len(self.example_text), self.hidden_dimension]
        )
        assert (
            pooled_output.shape == expected_shape
        ), f"Unexpected shape ({pooled_output.shape}) from return_cls_embedding=True"

    def test_call_mean_pooling(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        last_hidden_state: torch.Tensor = embeddings.last_hidden_state

        pooled_output: torch.Tensor = self.embedding_layer.pool_embeddings(
            last_hidden_state, mean_pooling=True, return_cls_embedding=False
        )

        call_pooled_output: torch.Tensor = self.embedding_layer(
            self.example_text, mean_pooling=True, return_cls_embedding=False
        )

        assert torch.equal(
            pooled_output, call_pooled_output
        ), f"__call__ method incorrectly calculated the pooled_output."

    def test_call_return_cls_embedding(self):
        tokens: BatchEncoding = self.embedding_layer.tokenize_text(self.example_text)

        embeddings: BaseModelOutput = self.embedding_layer.generate_embeddings(tokens)

        last_hidden_state: torch.Tensor = embeddings.last_hidden_state

        pooled_output: torch.Tensor = self.embedding_layer.pool_embeddings(
            last_hidden_state, mean_pooling=False, return_cls_embedding=True
        )

        call_pooled_output: torch.Tensor = self.embedding_layer(
            self.example_text, mean_pooling=False, return_cls_embedding=True
        )

        assert torch.equal(
            pooled_output, call_pooled_output
        ), f"__call__ method incorrectly calculated the pooled_output."
