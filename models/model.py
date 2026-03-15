import torch
import torch.nn as nn

from src.config import cfg
from src.data.crf import forward_alg, forward_calc, score_sentences, viterbi_algo
from src.data.lstm import get_lstm_features, get_neg_log_likelihood
from src.data.weights_embeds import init_embedding, init_linear, init_lstm
from src.training.losses import multiclass_focal_loss


class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tag_to_ix,
        embedding_dim,
        hidden_dim,
        word_lstm_layers=2,
        char_to_ix=None,
        pre_word_embeds=None,
        char_out_dimension=25,
        char_batch_norm=True,
        char_window_size=5,
        char_embedding_dim=25,
        char_hidden_dim=None,
        char_padding_idx=0,
        use_gpu=False,
        use_crf=True,
        char_mode="CNN",
        dropout=0.5,
        focal_loss_enabled=True,
        focal_loss_gamma=2.0,
        focal_loss_weight=1.0,
        entity_to_ix=None,
        sentence_entity_enabled=True,
        sentence_entity_hidden_dim=128,
        sentence_entity_pooling="mean_max",
        sentence_entity_loss_weight=0.2,
        sentence_entity_threshold=0.5,
        start_tag=cfg.START_TAG,
        stop_tag=cfg.STOP_TAG,
    ):
        super().__init__()

        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_lstm_layers = word_lstm_layers
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_batch_norm_enabled = char_batch_norm
        self.char_window_size = char_window_size
        self.char_mode = char_mode
        self.char_lstm_dim = char_hidden_dim or cfg.model.char_lstm_dim
        self.focal_loss_enabled = focal_loss_enabled
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_weight = focal_loss_weight
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        self.entity_to_ix = entity_to_ix or {}
        self.sentence_entity_enabled = sentence_entity_enabled and bool(self.entity_to_ix)
        self.sentence_entity_pooling = sentence_entity_pooling
        self.sentence_entity_loss_weight = sentence_entity_loss_weight
        self.sentence_entity_threshold = sentence_entity_threshold

        if self.char_mode not in {"CNN", "LSTM"}:
            raise ValueError(f"Unsupported char mode: {self.char_mode}")
        if self.word_lstm_layers < 1:
            raise ValueError(f"word_lstm_layers must be positive, got {self.word_lstm_layers}")
        if char_to_ix is None:
            raise ValueError("char_to_ix is required for character encoder")
        if self.char_window_size < 1:
            raise ValueError(f"char_window_size must be positive, got {self.char_window_size}")
        if self.sentence_entity_pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported sentence entity pooling: {self.sentence_entity_pooling}")

        self.char_embeds = nn.Embedding(
            len(char_to_ix),
            char_embedding_dim,
            padding_idx=char_padding_idx,
        )
        init_embedding(self.char_embeds.weight)
        self.char_embeds.weight.data[char_padding_idx].zero_()

        if self.char_mode == "LSTM":
            self.char_lstm = nn.LSTM(
                char_embedding_dim,
                self.char_lstm_dim,
                num_layers=1,
                bidirectional=True,
            )
            init_lstm(self.char_lstm)
            self.char_batch_norm = None
            lstm_input_dim = embedding_dim + self.char_lstm_dim * 2
        else:
            self.char_cnn3 = nn.Conv2d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(self.char_window_size, char_embedding_dim),
                padding=(self.char_window_size - 1, 0),
            )
            self.char_batch_norm = nn.BatchNorm2d(self.out_channels) if self.char_batch_norm_enabled else None
            lstm_input_dim = embedding_dim + self.out_channels

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.word_embeds.weight = nn.Parameter(torch.as_tensor(pre_word_embeds, dtype=torch.float32))
        else:
            init_embedding(self.word_embeds.weight)

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=self.word_lstm_layers,
            bidirectional=True,
            dropout=dropout if self.word_lstm_layers > 1 else 0.0,
        )
        init_lstm(self.lstm)

        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        init_linear(self.hidden2tag)
        self.register_buffer("focal_loss_alpha", torch.ones(self.tagset_size, dtype=torch.float32))

        if self.sentence_entity_enabled:
            sentence_input_dim = hidden_dim * 2 if self.sentence_entity_pooling in {"mean", "max"} else hidden_dim * 4
            self.sentence_entity_hidden = nn.Linear(sentence_input_dim, sentence_entity_hidden_dim)
            self.sentence_entity_output = nn.Linear(sentence_entity_hidden_dim, len(self.entity_to_ix))
            init_linear(self.sentence_entity_hidden)
            init_linear(self.sentence_entity_output)
        else:
            self.sentence_entity_hidden = None
            self.sentence_entity_output = None

        if self.use_crf:
            self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[self.start_tag], :] = -10000
            self.transitions.data[:, tag_to_ix[self.stop_tag]] = -10000

    def set_focal_loss_alpha(self, alpha):
        alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=self.focal_loss_alpha.device)
        if alpha_tensor.shape != self.focal_loss_alpha.shape:
            raise ValueError(
                f"Expected focal loss alpha with shape {tuple(self.focal_loss_alpha.shape)}, got {tuple(alpha_tensor.shape)}"
            )
        self.focal_loss_alpha.copy_(alpha_tensor)

    def compute_token_focal_loss(self, token_logits, tags, reduction):
        if not self.focal_loss_enabled:
            return None
        return multiclass_focal_loss(
            token_logits,
            tags,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha,
            reduction=reduction,
        )

    def pool_sentence_features(self, token_features):
        if token_features.ndim != 2:
            raise ValueError(f"Expected token features with shape [T, H], got {tuple(token_features.shape)}")

        if self.sentence_entity_pooling == "mean":
            return token_features.mean(dim=0)
        if self.sentence_entity_pooling == "max":
            return token_features.max(dim=0).values
        return torch.cat((token_features.mean(dim=0), token_features.max(dim=0).values), dim=0)

    def compute_sentence_entity_logits(self, token_features):
        if not self.sentence_entity_enabled:
            return None

        pooled = self.pool_sentence_features(token_features)
        hidden = torch.relu(self.sentence_entity_hidden(pooled))
        hidden = self.dropout(hidden)
        return self.sentence_entity_output(hidden)

    def predict_sentence_entities(self, token_features):
        logits = self.compute_sentence_entity_logits(token_features)
        if logits is None:
            return None, None

        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= self.sentence_entity_threshold
        return probabilities, predictions

    def predict_sentence_entities_from_inputs(self, sentence, chars, chars2_length, restore_order):
        _, token_features, _ = self._get_lstm_features(
            sentence,
            chars,
            chars2_length,
            restore_order,
            return_auxiliary=True,
        )
        return self.predict_sentence_entities(token_features)

    _score_sentence = score_sentences
    _get_lstm_features = get_lstm_features
    _forward_alg = forward_alg
    viterbi_decode = viterbi_algo
    neg_log_likelihood = get_neg_log_likelihood
    forward = forward_calc
