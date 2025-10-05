import torch
from torch.utils.data import DataLoader

from utils.tools import to_device, log
from model import FastSpeech2LossStage1, FastSpeech2LossStage2
from dataset import DatasetESDNeutral, DatasetESD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_evaluation(model, step, configs, dataset, criterion, forward_slice=slice(2, None), loss_labels=None):
    preprocess_config, model_config, train_config = configs

    batch_size = train_config["optimizer"]["batch_size"]
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    total_losses = None
    last_batch = None
    last_outputs = None

    for batch_group in data_loader:
        for batch in batch_group:
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(*batch[forward_slice])
                losses = criterion(batch, outputs)

            if total_losses is None:
                total_losses = [0.0 for _ in range(len(losses))]

            effective_batch_size = len(batch[0])
            for i, loss_val in enumerate(losses):
                total_losses[i] += loss_val.item() * effective_batch_size

            last_batch = batch
            last_outputs = outputs

    avg_losses = [v / len(dataset) for v in total_losses]

    if loss_labels is None:
        loss_labels = [f"Loss{i}" for i in range(len(avg_losses))]
    loss_report = ", ".join([f"{name}: {val:.4f}" for name, val in zip(["Total"] + loss_labels[1:], avg_losses)])
    message = f"Validation Step {step}, {loss_report}"
    return message


def evaluatestage1(model, step, configs):
    preprocess_config, model_config, train_config = configs
    dataset = DatasetESDNeutral("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    criterion = FastSpeech2LossStage1(preprocess_config, model_config).to(device)
    loss_labels = [
        "Total Loss",
        "Mel Loss",
        "Mel PostNet Loss",
        "Duration Loss",
    ]
    return _run_evaluation(model, step, configs, dataset, criterion, forward_slice=slice(2, None), loss_labels=loss_labels)


def evaluatestage2(model, step, configs):
    preprocess_config, model_config, train_config = configs
    dataset = DatasetESD("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    criterion = FastSpeech2LossStage2(preprocess_config, model_config).to(device)
    loss_labels = [
        "Total Loss",
        "Mel Loss",
        "Mel PostNet Loss",
        "Pitch Loss",
        "Energy Loss",
        "Duration Loss",
        "Emotion Loss",
        "Speaker Loss",
    ]
    # Stage 2 forward excludes the last two labels (emotion and speaker IDs)
    return _run_evaluation(model, step, configs, dataset, criterion, forward_slice=slice(2, -2), loss_labels=loss_labels)
