import torch
import torch.nn.functional as F


def build_focal_loss_alpha(
    train_data: list[dict[str, list[int]]],
    num_classes: int,
    alpha_power: float,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for record in train_data:
        tags = record.get("tags", [])
        if not tags:
            continue
        counts += torch.bincount(torch.as_tensor(tags, dtype=torch.long), minlength=num_classes).to(torch.float32)

    valid_mask = counts > 0
    if not torch.any(valid_mask):
        return torch.ones(num_classes, dtype=torch.float32)

    alpha = torch.zeros(num_classes, dtype=torch.float32)
    alpha[valid_mask] = counts[valid_mask].pow(-alpha_power)
    alpha[valid_mask] *= valid_mask.sum() / alpha[valid_mask].sum()
    return alpha


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets with shape [N], got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"Batch size mismatch between logits and targets: {logits.size(0)} != {targets.size(0)}"
        )

    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1 - pt).pow(gamma)) * log_pt

    if alpha is not None:
        if alpha.ndim != 1 or alpha.size(0) != logits.size(1):
            raise ValueError(f"Expected alpha with shape [{logits.size(1)}], got {tuple(alpha.shape)}")
        loss = loss * alpha.to(device=logits.device, dtype=logits.dtype)[targets]

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
