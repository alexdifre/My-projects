import torch
import numpy as np
import sys
import time
import torch.nn.functional as F

from CONTR_DISTILL.CKA.cka import cka # type: ignore

def adjust_learning_rate(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_with_crd(epoch, train_loader, module_list, criterion_list, optimizer, opt, diag=None):
    """
    Training function for one epoch using Contrastive Representation Distillation (CRD).

    Args:
        epoch (int): The current epoch number.
        train_loader (DataLoader): The data loader for the training set.
        module_list (list): A list of modules, where:
            - module_list[0] is the student model.
            - module_list[-1] is the teacher model.
        criterion_list (list): A list of loss functions, where:
            - criterion_list[0] is the Cross-Entropy loss criterion.
            - criterion_list[-1] is the CRD loss criterion.
        optimizer (Optimizer): The optimizer for the student model.
        opt (Namespace): An object containing training options (e.g., opt.beta, opt.print_freq).
        diag (DiagnosticTool, optional): A tool for logging diagnostics. Defaults to None.

    Returns:
        tuple: A tuple containing the average CKA score, average total loss, 
               and average cosine similarity for the epoch.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = module_list[0]
    teacher_model = module_list[-1]
    
    criterion_ce = criterion_list[0]
    criterion_crd = criterion_list[-1]

    # Set model modes correctly once
    student_model.train()
    teacher_model.eval()

    # --- Meter Initialization ---
    batch_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    cka_meter = AverageMeter()
    cos_sim_meter = AverageMeter()

    end = time.time()
    for idx, (input_data, target, index, contrast_idx) in enumerate(train_loader):
        
        # ====== Data Preparation ======
        input_data = input_data.float().to(device)
        target = target.to(device)
        index = index.to(device)
        contrast_idx = contrast_idx.to(device)

        # ====== Forward Pass ======
        # Student forward pass
        feat_s, logits_s = student_model(input_data, is_feat=True, preact=True)

        # Teacher forward pass (no gradients needed)
        with torch.no_grad():
            feat_t, _ = teacher_model(input_data, is_feat=True, preact=True)
            feat_t = [f.detach() for f in feat_t]

        # Extract features for distillation
        f_s_penult = feat_s[-2].view(feat_s[-2].size(0), -1)
        f_t_penult = feat_t[-2].view(feat_t[-2].size(0), -1)
        f_s_last = feat_s[-1]
        f_t_last = feat_t[-1]

        # ====== Loss Calculation ======
        loss_crd_penult = criterion_crd.forward_penultimate(f_s_penult, f_t_penult, index, contrast_idx)
        loss_crd_last = criterion_crd.forward_last(f_s_last, f_t_last, index, contrast_idx)
        loss_crd = 0.5 * loss_crd_penult + 0.5 * loss_crd_last
        
        loss_ce = criterion_ce(logits_s, target)

        # NOTE: The original code only used loss_crd. 
        # The standard approach is to combine both losses.
        # Uncomment the second line to include the classification loss.
        total_loss = opt.beta * loss_crd 
        # total_loss = loss_ce + opt.beta * loss_crd

        # ====== Backward Pass & Optimization ======
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ====== Update Meters and Logging ======
        loss_meter.update(total_loss.item(), input_data.size(0))
        batch_time_meter.update(time.time() - end)
        end = time.time()

        # Calculate metrics within no_grad to save memory and compute
        with torch.no_grad():
            cka_last = cka(f_s_last, f_t_last)
            
            # Project features to embedding space for cosine similarity
            f_s_embed = criterion_crd.embed_s_last(f_s_last)
            f_t_embed = criterion_crd.embed_t_last(f_t_last)
            cos_sim = F.cosine_similarity(f_s_embed, f_t_embed, dim=1).mean().item()
            
            cka_meter.update(cka_last, input_data.size(0))
            cos_sim_meter.update(cos_sim, input_data.size(0))

        if diag is not None:
            diag.log_step(optimizer)

        # Print logs
        if (idx + 1) % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Time {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                  f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                  f'CKA {cka_meter.avg:.2f}\t'
                  f'CosSim {cos_sim_meter.avg:.3f}')
            sys.stdout.flush()
            
    return cka_meter.avg, loss_meter.avg, cos_sim_meter.avg
    


def train_logits(epoch, train_loader, module_list, criterion_list, optimizer, scheduler, opt):
    """
    Train the student model on logits from the teacher and ground truth labels.
    """
    criterion_KL = criterion_list[1]
    criterion_ce = criterion_list[0]

    model_s = module_list[0]
    model_t = module_list[-1]
    
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        # In this function, we only need input and target.
        # index and contrast_idx are ignored.
        input, target, _, _ = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ====== Forward Pass ======
        # Get student features and logits
        _, logit_s = model_s(input, is_feat=True, preact=True)
        
        # Get teacher logits (no gradients needed)
        with torch.no_grad():
            _, logit_t = model_t(input, is_feat=True, preact=True)

        # ====== Calculate Losses ======
        # loss_distill: How well the student mimics the teacher's predictions
        loss_distill = criterion_KL(logit_s, logit_t)
        
        # loss_gt: How well the student predicts the true labels
        loss_gt = criterion_ce(logit_s, target)
        
        # Combine losses with a weighting factor (alpha)
        # opt.alpha could be a parameter, e.g., 0.7
        alpha = 0.7 
        loss = alpha * loss_distill + (1 - alpha) * loss_gt
        
        # ====== Metrics ======
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ====== Backward Pass & Optimization ======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update timers
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.avg:.3f}\t'
                  f'Loss {losses.avg:.4f}\t'
                  f'Acc@1 {top1.avg:.3f}\t'
                  f'Acc@5 {top5.avg:.3f}')
            sys.stdout.flush()

    # ====== End of Epoch ======
    # Step the scheduler ONCE per epoch, after all batches are processed.
    scheduler.step()
    print(f' * Epoch {epoch} finished. Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}. '
          f'LR updated to {scheduler.get_last_lr()[0]:.6f}')

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            input, target = batch[:2]
            
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
    
def validate_teacher(val_loader, model, opt):
  
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            input, target = batch[:2]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)

            # measure accuracy 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, 
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
