import os
import torch
import torch.nn as nn
import torch.optim as optim

import tensorboard_logger as tb_logger
from CONTR_DISTILL.training.training_funct import validate

def freeze_except(model, layers_to_train=("layer3", "layer4", "fc")):
    for name, param in model.named_parameters():
        if not any(layer in name for layer in layers_to_train):
            param.requires_grad = False

def fine_tune_teacher(model_t, train_loader, val_loader, opt):
    print("==> Fine-tuning teacher model...")

    freeze_except(model_t, layers_to_train=("layer3", "layer4", "fc"))  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model_t.parameters()),
        lr=opt.learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_ft)

    # logger for fine-tuning
    ft_logdir = os.path.join(opt.tb_folder, "teacher_finetune")
    os.makedirs(ft_logdir, exist_ok=True)
    logger = tb_logger.Logger(logdir=ft_logdir, flush_secs=2)

    best_acc = 0
    model_t = model_t.cuda()

    for epoch in range(opt.epochs_ft):
        model_t.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets, *_ in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model_t(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_acc = 100. * correct / total
        val_acc, _, val_loss = validate(val_loader, model_t, criterion, opt)

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', total_loss / len(train_loader), epoch)
        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('val_loss', val_loss, epoch)

        print(f"Epoch [{epoch+1}/{opt.epochs_ft}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(opt.teacher_path, f'{model_t.__class__.__name__}_finetuned_best.pth')
            torch.save(model_t.state_dict(), path)
            print("Saved fine-tuned teacher")

    print(f"Best fine-tuned teacher accuracy: {best_acc:.2f}")

