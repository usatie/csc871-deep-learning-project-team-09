import torch
from transformer.model.transformer import (
    subsequent_mask,
    attention,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
    Embeddings,
    Generator,
)
from transformer.utils.training import LabelSmoothing
from transformer.data.dataset import (
    tokenize,
    make_collate_fn,
)
import spacy
import unittest
from torchtext.vocab import Vocab
from collections import Counter


class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.d_model = 512
        self.d_ff = 2048
        self.h = 8
        self.dropout = 0.1
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.batch_size = 2
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_subsequent_mask(self):
        """Test the subsequent mask creation."""
        mask = subsequent_mask(5)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.all(mask == expected))

    def test_attention(self):
        """Test the attention mechanism."""
        batch_size = 2
        seq_len = 3
        d_k = 4

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        output, attn = attention(query, key, value)

        self.assertEqual(output.shape, (batch_size, seq_len, d_k))
        self.assertEqual(attn.shape, (batch_size, seq_len, seq_len))
        self.assertTrue(
            torch.allclose(attn.sum(dim=-1), torch.ones(batch_size, seq_len))
        )

    def test_multi_headed_attention(self):
        """Test the multi-headed attention mechanism."""
        mha = MultiHeadedAttention(self.h, self.d_model, self.dropout)

        batch_size = 2
        seq_len = 3

        query = torch.randn(batch_size, seq_len, self.d_model)
        key = torch.randn(batch_size, seq_len, self.d_model)
        value = torch.randn(batch_size, seq_len, self.d_model)

        output = mha(query, key, value)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_positionwise_feed_forward(self):
        """Test the position-wise feed forward network."""
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = ff(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_positional_encoding(self):
        """Test the positional encoding."""
        pe = PositionalEncoding(self.d_model, self.dropout)

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = pe(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_embeddings(self):
        """Test the embeddings layer."""
        emb = Embeddings(self.d_model, self.src_vocab_size)

        batch_size = 2
        seq_len = 3
        x = torch.randint(0, self.src_vocab_size, (batch_size, seq_len))

        output = emb(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_generator(self):
        """Test the generator layer."""
        gen = Generator(self.d_model, self.tgt_vocab_size)

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = gen(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.tgt_vocab_size))
        self.assertTrue(
            torch.allclose(output.exp().sum(dim=-1), torch.ones(batch_size, seq_len))
        )

    def test_encoder_layer(self):
        """Test a single encoder layer."""
        self_attn = MultiHeadedAttention(self.h, self.d_model, self.dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        encoder_layer = EncoderLayer(self.d_model, self_attn, ff, self.dropout)

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, self.d_model)
        mask = torch.ones(batch_size, 1, seq_len)

        output = encoder_layer(x, mask)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_decoder_layer(self):
        """Test a single decoder layer."""
        self_attn = MultiHeadedAttention(self.h, self.d_model, self.dropout)
        src_attn = MultiHeadedAttention(self.h, self.d_model, self.dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        decoder_layer = DecoderLayer(
            self.d_model, self_attn, src_attn, ff, self.dropout
        )

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, self.d_model)
        memory = torch.randn(batch_size, seq_len, self.d_model)
        src_mask = torch.ones(batch_size, 1, seq_len)
        tgt_mask = subsequent_mask(seq_len).unsqueeze(0)

        output = decoder_layer(x, memory, src_mask, tgt_mask)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_label_smoothing(self):
        """Test the label smoothing loss."""
        size = 10
        padding_idx = 0
        smoothing = 0.1
        criterion = LabelSmoothing(size, padding_idx, smoothing)

        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, size)
        target = torch.randint(1, size, (batch_size, seq_len))

        # Ensure x has the correct shape for label smoothing
        x = x.view(-1, size)
        target = target.view(-1)

        loss = criterion(x, target)

        self.assertIsInstance(loss.item(), float)
        # The loss should be negative because we're using log probabilities
        self.assertLess(loss.item(), 0)

    def test_tokenize(self):
        """Test the tokenization function."""
        nlp = spacy.load("en_core_web_sm")
        text = "Hello, world! This is a test."

        tokens = tokenize(text, nlp)

        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_make_collate_fn(self):
        """Test the collate function."""
        nlp = spacy.load("en_core_web_sm")
        tokenizers = {"en": nlp}

        # Create a vocabulary with default index
        counter = Counter(["hello", "world", "test", "sentence"])
        specials = ["<s>", "</s>", "<blank>", "<unk>"]
        for tok in specials:
            counter.update(tok)

        # Create vocabulary with special tokens
        vocab = Vocab(counter)

        collate_fn = make_collate_fn(
            type("Config", (), {"src_lang": "en", "tgt_lang": "en"}),
            tokenizers,
            vocab,
            vocab,
            self.device,
        )

        batch = [("hello world", "hello world"), ("test sentence", "test sentence")]
        src_tensor, tgt_tensor = collate_fn(batch)

        self.assertIsInstance(src_tensor, torch.Tensor)
        self.assertIsInstance(tgt_tensor, torch.Tensor)
        self.assertEqual(src_tensor.shape[0], len(batch))
        self.assertEqual(tgt_tensor.shape[0], len(batch))

    def test_make_collate_fn_german(self):
        """Test the collate function with German tokenizer."""
        nlp_de = spacy.load("de_core_news_sm")
        nlp_en = spacy.load("en_core_web_sm")
        tokenizers = {"de": nlp_de, "en": nlp_en}

        # Create vocabularies for both languages
        de_counter = Counter(["hallo", "welt", "test", "satz"])
        en_counter = Counter(["hello", "world", "test", "sentence"])
        specials = ["<s>", "</s>", "<blank>", "<unk>"]

        for tok in specials:
            de_counter.update(tok)
            en_counter.update(tok)

        vocab_de = Vocab(de_counter)
        vocab_en = Vocab(en_counter)

        # Create collate function
        collate_fn = make_collate_fn(
            type("Config", (), {"src_lang": "de", "tgt_lang": "en"}),
            tokenizers,
            vocab_de,
            vocab_en,
            self.device,
        )

        # Create a batch of German-English pairs
        batch = [("hallo welt", "hello world"), ("test satz", "test sentence")]

        # Test collate function
        src_tensor, tgt_tensor = collate_fn(batch)

        # Verify outputs
        self.assertIsInstance(src_tensor, torch.Tensor)
        self.assertIsInstance(tgt_tensor, torch.Tensor)
        self.assertEqual(src_tensor.shape[0], len(batch))
        self.assertEqual(tgt_tensor.shape[0], len(batch))

        # Verify that the tensors contain valid indices
        self.assertTrue(torch.all(src_tensor >= 0))
        self.assertTrue(torch.all(tgt_tensor >= 0))

        # Verify that the tensors have the correct shape
        self.assertEqual(src_tensor.dim(), 2)  # [batch_size, seq_len]
        self.assertEqual(tgt_tensor.dim(), 2)  # [batch_size, seq_len]


if __name__ == "__main__":
    unittest.main()
