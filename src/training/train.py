from collections import defaultdict
from dataclasses import asdict
import logging
import time
from pathlib import Path

import torch

from models.model import BiLSTM_CRF
from src.checkpoints import extract_model_state_dict, load_checkpoint_artifact, load_model_state_dict, save_full_checkpoint
from src.data.mapping import CHAR_PAD_TOKEN
from src.training.eval import evaluating
from src.training.losses import build_focal_loss_alpha
from src.training.tensors import prepare_sequence_batch

logger = logging.getLogger(__name__)


def define_model(mapping, config):
    entity_to_ix = mapping.get("entity_to_id", {})
    model = BiLSTM_CRF(
        vocab_size=len(mapping["word_to_id"]),
        tag_to_ix=mapping["tag_to_id"],
        embedding_dim=config.model.word_dim,
        hidden_dim=config.model.word_lstm_dim,
        char_to_ix=mapping["char_to_id"],
        pre_word_embeds=mapping["word_embs"],
        char_out_dimension=config.model.char_cnn_channels,
        char_batch_norm=config.model.char_batch_norm,
        char_window_size=config.model.char_window_size,
        char_embedding_dim=config.model.char_embedding_dim,
        char_hidden_dim=config.model.char_lstm_dim,
        char_padding_idx=mapping["char_to_id"].get(CHAR_PAD_TOKEN, 0),
        use_gpu=config.hardware.use_gpu,
        use_crf=config.model.crf,
        char_mode=config.model.char_mode,
        dropout=config.training.dropout,
        focal_loss_enabled=config.training.focal_loss_enabled,
        focal_loss_gamma=config.training.focal_loss_gamma,
        focal_loss_weight=config.training.focal_loss_weight,
        entity_to_ix=entity_to_ix,
        sentence_entity_enabled=config.sentence_entities.enabled,
        sentence_entity_hidden_dim=config.sentence_entities.hidden_dim,
        sentence_entity_pooling=config.sentence_entities.pooling,
        sentence_entity_loss_weight=config.sentence_entities.loss_weight,
        sentence_entity_threshold=config.sentence_entities.threshold,
        start_tag=config.START_TAG,
        stop_tag=config.STOP_TAG,
    )
    logger.info("Model initialized")
    return model


def configure_focal_loss(model, train_data, mapping, config):
    if not config.training.focal_loss_enabled:
        logger.info("Focal loss disabled")
        return

    alpha = build_focal_loss_alpha(
        train_data,
        num_classes=len(mapping["tag_to_id"]),
        alpha_power=config.training.focal_loss_alpha_power,
    )
    model.set_focal_loss_alpha(alpha.to(config.device))

    id_to_tag = mapping.get("id_to_tag", {})
    alpha_summary = ", ".join(
        f"{id_to_tag.get(index, index)}={value:.3f}"
        for index, value in sorted(enumerate(alpha.tolist()), key=lambda item: -item[1])[:8]
    )
    logger.info(
        "Configured focal loss | gamma=%.2f weight=%.2f alpha_power=%.2f top_alpha=[%s]",
        config.training.focal_loss_gamma,
        config.training.focal_loss_weight,
        config.training.focal_loss_alpha_power,
        alpha_summary,
    )


def build_optimizer(model, config):
    base_lr = config.training.learning_rate
    weight_decay = config.training.weight_decay
    embedding_lr = base_lr * config.training.embedding_learning_rate_scale

    other_parameters = []
    embedding_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("word_embeds."):
            embedding_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = [{"params": other_parameters, "lr": base_lr, "weight_decay": weight_decay}]
    # Pretrained embeddings are easier to overfit; a smaller LR keeps their signal stable.
    if embedding_parameters:
        parameter_groups.append({"params": embedding_parameters, "lr": embedding_lr, "weight_decay": 0.0})

    optimizer_name = config.training.optimizer_name
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameter_groups, lr=base_lr, momentum=config.training.momentum)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameter_groups, lr=base_lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameter_groups, lr=base_lr, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameter_groups, lr=base_lr, momentum=config.training.momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, config):
    if config.training.scheduler_name == "none":
        return None

    if config.training.scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.training.scheduler_factor,
            patience=1,
            threshold=config.training.scheduler_threshold,
            threshold_mode="abs",
            min_lr=config.training.scheduler_min_lr,
        )

    raise ValueError(f"Unsupported scheduler: {config.training.scheduler_name}")


def maybe_step_scheduler(scheduler, metric_value, optimizer, epoch, step):
    if scheduler is None:
        return

    previous_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler.step(metric_value)
    updated_lrs = [group["lr"] for group in optimizer.param_groups]

    if updated_lrs != previous_lrs:
        logger.info(
            "Scheduler updated learning rates | epoch=%d step=%d old_lrs=%s new_lrs=%s metric=%.4f",
            epoch,
            step,
            [f"{value:.6f}" for value in previous_lrs],
            [f"{value:.6f}" for value in updated_lrs],
            metric_value,
        )


def print_parameter_summary(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    for name, parameter in model.named_parameters():
        logger.info("%-50s | %-30s | %10s", name, str(tuple(parameter.shape)), f"{parameter.numel():,}")

    logger.info("--- By block ---")
    block_params = defaultdict(int)
    for name, parameter in model.named_parameters():
        block_name = name.split(".")[0]
        block_params[block_name] += parameter.numel()

    for block_name, parameter_count in sorted(block_params.items(), key=lambda item: -item[1]):
        logger.info("%-30s | %10s", block_name, f"{parameter_count:,}")


def save_loss_plot(losses, path):
    if not losses:
        return

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Logging step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Loss curve saved to %s", path)


def load_initial_weights_if_needed(model, config):
    if not config.training.weights:
        return

    weights_path = Path(config.training.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = load_checkpoint_artifact(weights_path, config.device)
    state_dict = extract_model_state_dict(checkpoint)
    load_model_state_dict(model, state_dict, context=str(weights_path))
    logger.info("Loaded initial weights from %s", weights_path)


def restore_best_checkpoint_if_available(model, config):
    if not config.model_path.exists():
        return

    checkpoint = load_checkpoint_artifact(config.model_path, config.device)
    state_dict = extract_model_state_dict(checkpoint)
    load_model_state_dict(model, state_dict, context=str(config.model_path))
    logger.info("Restored best checkpoint from %s", config.model_path)


def run_full_evaluation(model, train_data, dev_data, mapping, config, best_scores):
    best_train_F, best_dev_F = best_scores

    model.eval()
    best_train_F, train_metrics, _ = evaluating(model, train_data, best_train_F, mapping, config, "Train")
    best_dev_F, dev_metrics, save = evaluating(model, dev_data, best_dev_F, mapping, config, "Validation")

    if save:
        logger.info("Saving full checkpoint to %s", config.model_path)
        save_full_checkpoint(
            model,
            mapping,
            config,
            metadata={
                "train_metrics": asdict(train_metrics),
                "dev_metrics": asdict(dev_metrics),
                "best_train_f1": best_train_F,
                "best_dev_f1": best_dev_F,
            },
        )

    logger.info(
        "Metrics summary | train_f1=%.4f train_tok_acc=%.4f | dev_f1=%.4f dev_tok_acc=%.4f",
        train_metrics.f1,
        train_metrics.token_accuracy,
        dev_metrics.f1,
        dev_metrics.token_accuracy,
    )
    model.train()

    return best_train_F, best_dev_F, dev_metrics.f1, save


def train(model, train_data, dev_data, mapping, config):
    if not train_data:
        raise ValueError("Training data is empty")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    losses = []
    running_loss = 0.0
    best_dev_F = -1.0
    best_train_F = -1.0
    step = 0

    model.to(config.device)
    load_initial_weights_if_needed(model, config)
    configure_focal_loss(model, train_data, mapping, config)
    print_parameter_summary(model)

    started_at = time.time()
    logger.info(
        "Training started | epochs=%d optimizer=%s scheduler=%s lr=%.5f emb_lr_scale=%.3f "
        "momentum=%.3f weight_decay=%.6f char_mode=%s char_batch_norm=%s char_window=%d "
        "crf=%s focal=%s gamma=%.2f focal_weight=%.2f",
        config.training.epoch,
        config.training.optimizer_name,
        config.training.scheduler_name,
        config.training.learning_rate,
        config.training.embedding_learning_rate_scale,
        config.training.momentum,
        config.training.weight_decay,
        config.model.char_mode,
        config.model.char_batch_norm,
        config.model.char_window_size,
        config.model.crf,
        config.training.focal_loss_enabled,
        config.training.focal_loss_gamma,
        config.training.focal_loss_weight,
    )

    for epoch in range(1, config.training.epoch + 1):
        model.train()
        logger.info("Epoch %d/%d started", epoch, config.training.epoch)
        for index in torch.randperm(len(train_data)).tolist():
            step += 1
            data = train_data[index]
            batch = prepare_sequence_batch(data, config.model.char_mode)

            optimizer.zero_grad(set_to_none=True)

            words = batch.words.to(config.device)
            tags = batch.tags.to(config.device)
            chars = batch.chars.to(config.device)
            entity_targets = batch.entity_targets.to(config.device) if batch.entity_targets is not None else None

            neg_log_likelihood = model.neg_log_likelihood(
                words,
                tags,
                chars,
                batch.char_lengths,
                batch.restore_order,
                entity_targets=entity_targets,
            )
            running_loss += neg_log_likelihood.item() / max(1, len(data["words"]))
            neg_log_likelihood.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()

            if step % config.eval.plot_every == 0:
                averaged_loss = running_loss / config.eval.plot_every
                logger.info("Epoch=%d Step=%d Loss=%.6f", epoch, step, averaged_loss)
                losses.append(averaged_loss)
                running_loss = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d completed | lr=%.6f", epoch, current_lr)
        best_train_F, best_dev_F, current_dev_f1, _ = run_full_evaluation(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            mapping=mapping,
            config=config,
            best_scores=(best_train_F, best_dev_F),
        )
        maybe_step_scheduler(scheduler, current_dev_f1, optimizer, epoch, step)

    elapsed = time.time() - started_at
    restore_best_checkpoint_if_available(model, config)
    logger.info(
        "Training finished in %.2fs | best_train_f1=%.4f best_dev_f1=%.4f",
        elapsed,
        best_train_F,
        best_dev_F,
    )
    save_loss_plot(losses, config.paths.loss_plot_file)
