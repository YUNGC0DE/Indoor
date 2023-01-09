import os

import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train(net, train_loader, optimizer, device, logger, epoch, args):

    net.train()
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = net(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        if i % args.log_freq == 0:
            logger.report_scalar("Loss", "Train", iteration=epoch * len(train_loader) + i, value=loss_value)
        losses.backward()
        optimizer.step()


best_metric = -1


def eval(net, val_loader, lr_scheduler, model_path, device, logger, epoch):
    global best_metric
    metric = MeanAveragePrecision()
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predicts = net(images)
            metric.update(predicts, targets)
        metrics = metric.compute()

        for key, value in metrics.items():
            logger.report_scalar("mAP", key, iteration=epoch, value=value)

        map_50 = metrics["map_50"]
        reduced_lr = lr_scheduler.step(map_50)

        if map_50 > best_metric:
            best_metric = map_50
            state = {
                "val_metric": best_metric,
                "epoch": epoch,
                "net": net.__class__.__name__,
                "net_state": net.state_dict(),
            }
            torch.save(
                state,
                os.path.join(model_path, f"model_best.pkl"),
            )
            print("Saved new best model")
        return reduced_lr


def train_net(net, train_loader, val_loader, optimizer, lr_scheduler, device, logger, model_path, args):
    for i in range(args.epochs):
        logger.report_scalar("Lr", "", iteration=i, value=optimizer.param_groups[0]["lr"])
        train(net, train_loader, optimizer, device, logger, i, args)
        reduced = eval(net, val_loader, lr_scheduler, model_path, device, logger, i)
        if reduced:
            print("Lr has been reduced")
            best_model_path = os.path.join(model_path, f"model_best.pkl")
            model_weights = torch.load(best_model_path)
            net.load_state_dict(model_weights["net_state"])

