import torch


def score_sentences(self, feats, tags):
    indices = torch.arange(feats.size(0), device=feats.device)
    start_tensor = torch.tensor([self.tag_to_ix[self.start_tag]], dtype=torch.long, device=feats.device)
    stop_tensor = torch.tensor([self.tag_to_ix[self.stop_tag]], dtype=torch.long, device=feats.device)

    pad_start_tags = torch.cat([start_tensor, tags])
    pad_stop_tags = torch.cat([tags, stop_tensor])

    return torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[indices, tags])


def forward_alg(self, feats):
    forward_var = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
    forward_var[0][self.tag_to_ix[self.start_tag]] = 0.0

    for feat in feats:
        emit_score = feat.view(-1, 1)
        tag_var = forward_var + self.transitions + emit_score
        forward_var = torch.logsumexp(tag_var, dim=1).view(1, -1)

    terminal_var = (forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]).view(1, -1)
    return torch.logsumexp(terminal_var, dim=1).squeeze()


def viterbi_algo(self, feats):
    backpointers = []

    forward_var = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
    forward_var[0][self.tag_to_ix[self.start_tag]] = 0.0

    for feat in feats:
        next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
        best_scores, best_paths = torch.max(next_tag_var, dim=1)
        forward_var = best_scores + feat
        backpointers.append(best_paths.tolist())

    terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
    terminal_var[self.tag_to_ix[self.stop_tag]] = -10000.0
    terminal_var[self.tag_to_ix[self.start_tag]] = -10000.0

    best_tag_id = torch.argmax(terminal_var).item()
    path_score = terminal_var[best_tag_id]

    best_path = [best_tag_id]
    for backpointer in reversed(backpointers):
        best_tag_id = backpointer[best_tag_id]
        best_path.append(best_tag_id)

    start_tag_id = best_path.pop()
    assert start_tag_id == self.tag_to_ix[self.start_tag]

    best_path.reverse()
    return path_score, best_path


def forward_calc(self, sentence, chars, chars2_length, restore_order):
    feats = self._get_lstm_features(sentence, chars, chars2_length, restore_order)

    if self.use_crf:
        score, tag_seq = self.viterbi_decode(feats)
    else:
        score, tag_seq = torch.max(feats, 1)
        tag_seq = tag_seq.tolist()

    return score, tag_seq
