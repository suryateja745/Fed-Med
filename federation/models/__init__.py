"""
Neural network architectures, loss functions, and evaluation metrics for FedMed.
"""

from federation.models.metrics import (
    PyTorchDiceCELoss,
    PyTorchDiceLoss,
    compute_dice_score,
    get_loss_function,
)
from federation.models.trainer import (
    get_lr_scheduler,
    get_optimizer,
    train_epoch,
    train_local_client,
    validate,
)
from federation.models.unet3d import (
    FedMedUNet3D,
    PyTorchUNet3D,
    build_unet3d_from_config,
    count_parameters,
    create_unet3d_model,
    get_model_parameters,
    initialize_weights,
    load_model_checkpoint,
    save_model_checkpoint,
    set_model_parameters,
)

__all__ = [
    # Model
    "FedMedUNet3D",
    "PyTorchUNet3D",
    "create_unet3d_model",
    "build_unet3d_from_config",
    "initialize_weights",
    "count_parameters",
    "get_model_parameters",
    "set_model_parameters",
    "save_model_checkpoint",
    "load_model_checkpoint",
    # Metrics & Losses
    "PyTorchDiceLoss",
    "PyTorchDiceCELoss",
    "get_loss_function",
    "compute_dice_score",
    # Training & Validation
    "get_optimizer",
    "get_lr_scheduler",
    "train_epoch",
    "validate",
    "train_local_client",
]