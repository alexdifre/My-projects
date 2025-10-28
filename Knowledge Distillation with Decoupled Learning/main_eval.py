import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from CONTR_DISTILL.main_functions.main_functions import *
from CONTR_DISTILL.analysis_results.analysis_results  import analysis_results


def main():
    """Main function to run the two-phase distillation process."""
    opt = parse_option()

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        cudnn.benchmark = True

    # --- Load Data ---
    train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        k=opt.nce_k,
        mode=opt.mode
    )
    opt.n_data = n_data
    n_cls = 100

    # --- Load Models ---
    print("==> Initializing models...")
    teacher_model = TeachResNet34(n_cls).to(device)
    student_model = model_dict[opt.model_s](num_classes=n_cls).to(device)


    # Base directory in Google Drive
    base_dir = r'C:\Users\Alessandro Di Frenna\OneDrive\Desktop\CONTR_DISTILL_resnet32s_resnet34t\CONTR_DISTILL'

    # Teacher path
    teacher_path = os.path.join(
        base_dir, 
        'main_functions', 'teacher_models', 'TeachResNet34_finetuned_best.pth'
    )

    # Student path
    student_path = os.path.join(
        base_dir,
        'main_functions', 'student_model', 'S_resnet20T_ResNet34', 'resnet20_best.pth'
    )

    # Load models
    teacher_model.load_state_dict(torch.load(teacher_path))
    print(f"==> Loaded pre-trained teacher_model from {teacher_path}")

    checkpoint = torch.load(student_path)
    student_model.load_state_dict(checkpoint["model"])
    print(f"==> Loaded pre-trained student_model from {student_path}")

    #-------------ELABORATING RESULTS-------------------------
    y_true = get_y_true(val_loader)
    y_teacher = get_y_teacher(teacher_model, val_loader)
    y_student = get_y_student(student_model, val_loader)
    
    print('==> ELABORATING RESULTS...')
    STUDENT_NAME = "ResNet20"
    analysis_results(teacher_model, STUDENT_NAME, y_true, y_teacher, y_student)
    print('==> FINISH!')

if __name__ == '__main__':
    main()