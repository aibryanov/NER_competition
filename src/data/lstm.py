import torch
import torch.nn as nn


def encode_token_features(self, sentence, chars2, chars2_length, restore_order):
    if self.char_mode == "LSTM":
        chars_embeds = self.char_embeds(chars2).transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            chars_embeds,
            chars2_length,
            enforce_sorted=True,
        )
        lstm_out, _ = self.char_lstm(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = outputs.transpose(0, 1)

        chars_embeds_temp = torch.zeros(outputs.size(0), outputs.size(2), device=outputs.device)
        for index, output_length in enumerate(output_lengths.tolist()):
            chars_embeds_temp[index] = torch.cat(
                (
                    outputs[index, output_length - 1, : self.char_lstm_dim],
                    outputs[index, 0, self.char_lstm_dim :],
                )
            )

        chars_embeds = chars_embeds_temp.clone()
        for sorted_index, original_index in restore_order.items():
            chars_embeds[original_index] = chars_embeds_temp[sorted_index]

    elif self.char_mode == "CNN":
        chars_embeds = self.char_embeds(chars2).unsqueeze(1)
        chars_cnn_out = self.char_cnn3(chars_embeds)
        if self.char_batch_norm is not None:
            chars_cnn_out = self.char_batch_norm(chars_cnn_out)
        chars_embeds = nn.functional.max_pool2d(
            chars_cnn_out,
            kernel_size=(chars_cnn_out.size(2), 1),
        ).view(chars_cnn_out.size(0), self.out_channels)
    else:
        raise ValueError(f"Unsupported char mode: {self.char_mode}")

    embeds = self.word_embeds(sentence)
    embeds = torch.cat((embeds, chars_embeds), 1).unsqueeze(1)
    embeds = self.dropout(embeds)

    lstm_out, _ = self.lstm(embeds)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
    lstm_out = self.dropout(lstm_out)
    return lstm_out


def get_lstm_features(self, sentence, chars2, chars2_length, restore_order, return_auxiliary=False):
    token_features = encode_token_features(self, sentence, chars2, chars2_length, restore_order)
    token_logits = self.hidden2tag(token_features)
    sentence_entity_logits = self.compute_sentence_entity_logits(token_features)

    if return_auxiliary:
        return token_logits, token_features, sentence_entity_logits
    return token_logits


def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, restore_order, entity_targets=None):
    feats, _, sentence_entity_logits = self._get_lstm_features(
        sentence,
        chars2,
        chars2_length,
        restore_order,
        return_auxiliary=True,
    )

    if self.use_crf:
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        loss = forward_score - gold_score
    else:
        loss = nn.functional.cross_entropy(feats, tags)

    if self.sentence_entity_enabled and sentence_entity_logits is not None and entity_targets is not None:
        auxiliary_loss = nn.functional.binary_cross_entropy_with_logits(sentence_entity_logits, entity_targets)
        loss = loss + self.sentence_entity_loss_weight * auxiliary_loss

    return loss
