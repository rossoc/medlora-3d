import contextlib
import numpy as np
import torch
from tqdm.auto import tqdm
from monai.inferers.inferer import SlidingWindowInferer
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceCELoss


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, roi):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.5)
    losses = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = inferer(images, model)
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1, keepdim=True)  # type: ignore
        num_classes = int(logits.shape[1])  # type: ignore
        y_onehot = (
            torch.nn.functional.one_hot(
                labels.long().squeeze(1), num_classes=num_classes
            )
            .permute(0, 4, 1, 2, 3)
            .float()
        )
        p_onehot = (
            torch.nn.functional.one_hot(
                preds.long().squeeze(1), num_classes=num_classes
            )
            .permute(0, 4, 1, 2, 3)
            .float()
        )
        dice_metric(y_pred=p_onehot.to(device), y=y_onehot.to(device))
    return float(np.mean(losses) if losses else np.nan), float(
        dice_metric.aggregate().item()  # type: ignore
    )


def train(
    model,
    loaders,
    device,
    roi=(96, 96, 96),
    max_epochs=100,
    lr=1e-4,
    wd=1e-4,
    early_stopping=True,
    patience=10,
    min_epochs=10,
):
    train_loader, val_loader, train_eval_loader = loaders
    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True
    )  # monai.losses.DiceLoss(sigmoid=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_state, best_dice, best_counter = None, -1.0, 0
    tr_losses, va_losses, va_dices = [], [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        count = 0
        loss_avg = 0
        pbar = tqdm(
            train_loader,
            desc=f"[Training] epoch {epoch}/{max_epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in pbar:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            ctx = (
                torch.autocast("cuda", dtype=torch.float16)
                if device == "cuda"
                else contextlib.nullcontext()
            )
            with ctx:
                logits = model(images)
                loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            count += 1
            loss_avg += (float(loss.item()) - loss_avg) / count
            pbar.set_postfix(loss=f"{loss_avg:.4f}")
        tr_losses.append(loss_avg)

        eval_loss, eval_dice = evaluate(model, val_loader, loss_fn, device, roi)
        va_losses.append(eval_loss)
        va_dices.append(eval_dice)

        if eval_dice > best_dice:
            best_dice = eval_dice
            best_state = {
                k: v.cpu() for k, v in model.named_parameters() if v.requires_grad
            }
            best_counter = 0
        else:
            best_counter += 1
        if early_stopping and epoch >= min_epochs and best_counter >= patience:
            break

    _, tdice_eval = evaluate(model, train_eval_loader, loss_fn, device, roi)
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return {
        "train_losses": tr_losses,
        "val_losses": va_losses,
        "val_dices": va_dices,
        "best_val_dice": best_dice,
        "train_eval_dice": tdice_eval,
    }, best_state
