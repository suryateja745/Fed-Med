from pathlib import Path

from app.ml.trainer import dice_score, train_local_model


def test_dice_score_perfect_prediction() -> None:
    import torch

    predictions = torch.zeros(1, 2, 4, 4, 4)
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)

    predictions[:, 1] = 10.0
    targets[:] = 1

    score = dice_score(predictions, targets)

    assert score > 0.99


def test_local_training_creates_checkpoints(
    tmp_path: Path,
) -> None:
    result = train_local_model(
        epochs=1,
        batch_size=1,
        checkpoint_dir=tmp_path,
    )

    assert len(result["history"]) == 1
    assert 0.0 <= result["best_dice"] <= 1.0

    assert Path(
        result["latest_checkpoint"]
    ).exists()

    assert Path(
        result["best_checkpoint"]
    ).exists()


def test_local_training_history_contains_metrics(
    tmp_path: Path,
) -> None:
    result = train_local_model(
        epochs=1,
        batch_size=1,
        checkpoint_dir=tmp_path,
    )

    metrics = result["history"][0]

    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "dice" in metrics