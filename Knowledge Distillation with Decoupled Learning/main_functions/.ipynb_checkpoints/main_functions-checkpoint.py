import os
import argparse
import socket
import time

import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights


from CONTR_DISTILL.models.TeachResNet18 import TeachResNet18
from CONTR_DISTILL.models.TeachResNet34 import TeachResNet34
from CONTR_DISTILL.models.TeachResNet50 import TeachResNet50
from CONTR_DISTILL.models import model_dict
from CONTR_DISTILL.dataset.cifar100 import get_cifar100_dataloaders_sample
from CONTR_DISTILL.training.training_funct import adjust_learning_rate, train_with_crd, train_logits, validate, validate_teacher 
from CONTR_DISTILL.losses.CRD_loss import CRDLoss
from CONTR_DISTILL.losses.KL_loss import KL_loss
from CONTR_DISTILL.CKA.cka import cka # type: ignore
from CONTR_DISTILL.estimator.kraskov import kraskov_estimator
from CONTR_DISTILL.training.fine_tuning import fine_tune_teacher

from CONTR_DISTILL.diagnostics.DiagnosticsLogger import DiagnosticsLogger
from torch.utils.tensorboard import SummaryWriter

def parse_option():

    hostname = socket.gethostname()
 
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=14, help='num of workers to use')
    parser.add_argument('--epochs_first', type=int, default=50, help='number of training epochs first part')
    parser.add_argument('--beta', type=float, default=0.8, help='weight for CRD loss')
    parser.add_argument('--epochs_second', type=int, default=50, help='number of training epochs_second part')
    parser.add_argument('--init_epochs', type=int, default=10, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,60,85', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    # model
    parser.add_argument('--model_s', type=str, default='resnet32')

    # distillation
    parser.add_argument('--distill', type=str, default='crd', help='distillation method') 

    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # NCE distillation
    parser.add_argument('--feat_dim', default=256, type=int, help='feature dimension') # 128 default
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # teeacher fine-tuning
    parser.add_argument('--fine_tune_teacher', action='store_true', help='fine-tune the teacher model before distillation')
    parser.add_argument('--epochs_ft', type=int, default=30, help='epochs for fine-tuning the teacher')


    opt = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Ottieni la directory dello script
    
    opt.model_path = os.path.join(script_dir, "student_model")  # Percorso relativo alla directory dello script
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
        
    opt.tb_path = os.path.join(script_dir, "student_tensorboards") 
    if not os.path.isdir(opt.tb_path):
        os.makedirs(opt.tb_path)
    
    opt.teacher_path = os.path.join(script_dir, "teacher_models")
    if not os.path.isdir(opt.teacher_path):
        os.makedirs(opt.teacher_path)

    
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    opt.model_name = f"S_{opt.model_s}T_ResNet34"                                                               

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    segments = model_path.split('/')[-2].split('_')
    return segments[0]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model
    

def kraskov_ave(model_s, model_t, val_loader, device='cuda', n_batches=3):
    mi_scores = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        data = batch[0].to(device)
        with torch.no_grad():
            feat_s, _ = model_s(data, is_feat=True)
            feat_t, _ = model_t(data, is_feat=True)

        f_s = feat_s[-1].view(feat_s[-1].shape[0], -1).cpu().numpy()
        f_t = feat_t[-1].view(feat_t[-1].shape[0], -1).cpu().numpy()

        mi_scores.append(kraskov_estimator(f_s, f_t, k=5))

    return np.mean(mi_scores)


def compute_cka_on_batch(model_s, model_t, data, device='cuda'):
    with torch.no_grad():
        data = data.to(device)
        feat_s, _ = model_s(data, is_feat=True)
        feat_t, _ = model_t(data, is_feat=True)

        f_s = feat_s[-1].view(feat_s[-1].shape[0], -1) 
        f_t = feat_t[-1].view(feat_t[-1].shape[0], -1) 
        cka_score = cka(f_s, f_t)
        
    return cka_score


def get_one_batch(loader):
    """
    Retrieves the first batch from a DataLoader and returns the data tensor.

    Args:
        loader (torch.utils.data.DataLoader): The DataLoader to fetch from.

    Returns:
        torch.Tensor: The first batch of data, with a batch dimension ensured.
    """
    # Get the first batch from the iterator
    try:
        batch = next(iter(loader))
    except StopIteration:
        # This handles the case where the loader is empty
        return None

    # The data is typically the first element of the batch
    data = batch[0]

    # Ensure the data has a batch dimension
    if data.dim() == 3:
        data = data.unsqueeze(0)

    return data

# herper func to get outputs of the model 
def _get_model_outputs(model, loader, device='cuda', return_probs=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = batch[:2]
            inputs = inputs.to(device)

            _, out = model(inputs, is_feat=False)
            if return_probs:
                out = torch.softmax(out, dim=1)
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)

# true labels 
def get_y_true(loader):
    y_true = []
    for batch in loader:
        _, targets = batch[:2]
        y_true.append(targets)
    return torch.cat(y_true, dim=0)

# teacher outputs 
def get_y_teacher(model_t, loader, device='cuda', return_probs=False):
    return _get_model_outputs(model_t, loader, device=device, return_probs=return_probs)


# student outputs 
def get_y_student(model_s, loader, device='cuda', return_probs=False): 

    return _get_model_outputs(model_s, loader, device=device, return_probs=return_probs)


# module to match features
class LinearEmbed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)
