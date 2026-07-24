from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK


@dataclass(frozen=True)
class DistillationConfig:
    classification_weight: float = 0.5
    box_weight: float = 0.25
    temperature: float = 2.0
    topk: int = 1500


_TEACHER: torch.nn.Module | None = None
_CONFIG = DistillationConfig()
_STATS = {"batches": 0, "classification": 0.0, "box": 0.0, "total": 0.0}


def configure_distillation(teacher: torch.nn.Module, config: DistillationConfig) -> None:
    global _TEACHER, _CONFIG
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    _TEACHER = teacher
    _CONFIG = config


def reset_distillation_stats() -> None:
    _STATS.update({"batches": 0, "classification": 0.0, "box": 0.0, "total": 0.0})


def consume_distillation_stats() -> dict[str, float | int]:
    batches = int(_STATS["batches"])
    result = {
        "batches": batches,
        "classification": _STATS["classification"] / batches if batches else 0.0,
        "box": _STATS["box"] / batches if batches else 0.0,
        "total": _STATS["total"] / batches if batches else 0.0,
    }
    reset_distillation_stats()
    return result


def _raw_predictions(output):
    return output[1] if isinstance(output, tuple) else output


def distillation_losses(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = _CONFIG.temperature
    student_scores = student["scores"]
    teacher_scores = teacher["scores"].detach()
    teacher_confidence = teacher_scores.sigmoid().amax(dim=1)
    selected_count = min(_CONFIG.topk, teacher_confidence.shape[-1])
    selected = teacher_confidence.topk(selected_count, dim=-1).indices
    score_index = selected[:, None, :].expand(-1, student_scores.shape[1], -1)
    student_selected_scores = student_scores.gather(2, score_index) / temperature
    teacher_selected_scores = teacher_scores.gather(2, score_index) / temperature
    teacher_probability = teacher_selected_scores.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    student_probability = student_selected_scores.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    classification_kl = teacher_probability * (teacher_probability.log() - student_probability.log())
    classification_kl += (1.0 - teacher_probability) * (
        (1.0 - teacher_probability).log() - (1.0 - student_probability).log()
    )

    selected_confidence = teacher_confidence.gather(1, selected)
    anchor_weights = (selected_confidence + 0.05) / (selected_confidence.mean(dim=1, keepdim=True) + 0.05)
    classification_loss = (classification_kl.mean(dim=1) * anchor_weights).mean() * temperature**2

    batch_size, box_channels, anchor_count = student["boxes"].shape
    distribution_bins = box_channels // 4
    box_index = selected[:, None, None, :].expand(-1, 4, distribution_bins, -1)
    student_boxes = student["boxes"].view(batch_size, 4, distribution_bins, anchor_count).gather(3, box_index)
    teacher_boxes = teacher["boxes"].detach().view(batch_size, 4, distribution_bins, anchor_count).gather(3, box_index)
    teacher_distribution = F.softmax(teacher_boxes / temperature, dim=2)
    student_log_distribution = F.log_softmax(student_boxes / temperature, dim=2)
    box_kl = F.kl_div(student_log_distribution, teacher_distribution, reduction="none").sum(dim=2).mean(dim=1)
    box_loss = (box_kl * anchor_weights).mean() * temperature**2
    return classification_loss, box_loss


class DistillationDetectionModel(DetectionModel):
    """Detection model that adds teacher output KL terms during training only."""

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        supervised_loss, loss_items = self.criterion(preds, batch)
        if not self.training or _TEACHER is None:
            return supervised_loss, loss_items

        with torch.no_grad():
            teacher_predictions = _raw_predictions(_TEACHER(batch["img"]))
        classification_kd, box_kd = distillation_losses(_raw_predictions(preds), teacher_predictions)
        batch_size = batch["img"].shape[0]
        weighted_classification = _CONFIG.classification_weight * classification_kd
        weighted_box = _CONFIG.box_weight * box_kd
        loss = supervised_loss + (weighted_classification + weighted_box) * batch_size
        logged_items = loss_items.clone()
        logged_items[1] += weighted_classification.detach()
        logged_items[2] += weighted_box.detach()

        total = weighted_classification + weighted_box
        _STATS["batches"] += 1
        _STATS["classification"] += float(classification_kd.detach())
        _STATS["box"] += float(box_kd.detach())
        _STATS["total"] += float(total.detach())
        return loss, logged_items


class DistillationTrainer(DetectionTrainer):
    """Detection trainer that constructs a distillation-aware student model."""

    def get_model(self, cfg: str | dict | None = None, weights=None, verbose: bool = True):
        model = DistillationDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
