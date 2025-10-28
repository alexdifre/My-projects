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
from CONTR_DISTILL.analysis_results import analysis_results

def main():
    """Main function to run the two-phase distillation process."""
    opt = parse_option()
    best_acc = 0

    # --- Setup Loggers ---
    # Consolidate loggers into one directory for simplicity
    log_dir = os.path.join(opt.save_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    csv_path = os.path.join(log_dir, 'diagnostics.csv')
    print(f"Logging to {log_dir}")

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

    # Load or fine-tune teacher model
    teacher_name = teacher_model.__class__.__name__
    teacher_path = os.path.join(opt.teacher_path, f'{teacher_name}_finetuned_best.pth')
    if opt.fine_tune_teacher:
        print("==> Fine-tuning teacher model...")
        fine_tune_teacher(teacher_model, train_loader, val_loader, opt)
        teacher_model.load_state_dict(torch.load(teacher_path))
        print("==> Fine-tuned and loaded teacher model.")
    else:
        teacher_model.load_state_dict(torch.load(teacher_path))
        print(f"==> Loaded pre-trained teacher model from {teacher_path}")

    # Freeze teacher model completely
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    diag = DiagnosticsLogger(student_model, writer, csv_path=csv_path)

    # --- Setup Criteria and Optimizers ---
    print("==> Setting up criteria and optimizers...")
    # Get feature dimensions
    with torch.no_grad():
        images = get_one_batch(train_loader)
        feat_s, _ = student_model(images.to(device), is_feat=True, preact=True)
        feat_t, _ = teacher_model(images.to(device), is_feat=True, preact=True)
        opt.s_dim = feat_s[-2].view(feat_s[-2].size(0), -1).shape[1]
        opt.t_dim = feat_t[-2].view(feat_t[-2].size(0), -1).shape[1]
        opt.s_last_dim = feat_s[-1].shape[1]
        opt.t_last_dim = feat_t[-1].shape[1]
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = KL_loss(opt.kd_T)
    criterion_crd = CRDLoss(opt)
    
    criterion_list = nn.ModuleList([criterion_ce, criterion_kl, criterion_crd]).to(device)
    
    # Freeze teacher-side embedding models in CRD
    for param in criterion_crd.embed_t_penult.parameters():
        param.requires_grad = False
    for param in criterion_crd.embed_t_last.parameters():
        param.requires_grad = False
        
    # All modules that will be used in the training loops
    module_list = nn.ModuleList([
        student_model, criterion_crd.embed_s_penult, criterion_crd.embed_t_penult,
        criterion_crd.embed_s_last, criterion_crd.embed_t_last, teacher_model
    ]).to(device)

    # --- Phase 1: Train with Contrastive Distillation ---
    # Optimizer for the student model and its CRD embeddings
    optimizer_phase1 = optim.SGD(
        list(student_model.parameters()) + list(criterion_crd.embed_s_penult.parameters()) + list(criterion_crd.embed_s_last.parameters()),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    print("\n--- Starting Phase 1: Contrastive Feature Distillation ---")
    for epoch in range(1, opt.epochs_first + 1):
        adjust_learning_rate(epoch, opt, optimizer_phase1)
        start_time = time.time()
        
        cka_score, train_loss, cos_sim = train_with_crd(epoch, train_loader, module_list, criterion_list, optimizer_phase1, opt)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} summary: CKA={cka_score:.2f}, Loss={train_loss:.4f}, CosSim={cos_sim:.3f}, Time={epoch_time:.2f}s")

    writer.add_scalar('phase1/cosine_similarity', cos_sim, epoch)
    writer.add_scalar('phase1/cka', cka_score, epoch)
    writer.add_scalar('phase1/train_loss', train_loss, epoch)

    diag.close() # Close diagnostic logger after phase 1

    # --- Phase 2: Train with Logits Distillation (KL Loss) ---
    # In this phase, we only fine-tune the last layers
    for param in student_model.parameters():
        param.requires_grad = False # Freeze the entire student first
    for param in student_model.layer3.parameters():
        param.requires_grad = True # Unfreeze layer3
    for param in student_model.fc.parameters():
        param.requires_grad = True # Unfreeze fc

    optimizer_phase2 = optim.Adam([
        {'params': student_model.layer3.parameters(), 'lr': 1e-5},
        {'params': student_model.fc.parameters(), 'lr': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_phase2, gamma=0.98)

    print("\n--- Starting Phase 2: Logits Distillation & Fine-tuning ---")
    for epoch in range(1, opt.epochs_second + 1):
        start_time = time.time()
        
        train_acc, train_loss = train_logits(epoch, train_loader, module_list, criterion_list, optimizer_phase2, scheduler, opt)
        test_acc, test_acc_top5, test_loss = validate(val_loader, student_model, criterion_ce, opt)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} summary: TrainAcc={train_acc:.2f}, TestAcc={test_acc:.2f}, Time={epoch_time:.2f}s")
        
        # Log metrics
        writer.add_scalar('phase2/train_acc', train_acc, epoch)
        writer.add_scalar('phase2/train_loss', train_loss, epoch)
        writer.add_scalar('phase2/test_acc', test_acc, epoch)
        writer.add_scalar('phase2/test_loss', test_loss, epoch)
        writer.add_scalar('phase2/test_acc_top5', test_acc_top5, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f" New best accuracy: {best_acc:.2f}!")
            state = {'epoch': epoch, 'model': student_model.state_dict(), 'best_acc': best_acc}
            save_file = os.path.join(opt.save_folder, f'{opt.model_s}_best.pth')
            torch.save(state, save_file)

    # --- Final Actions ---
    writer.close()
    
    print("\n--- Training Complete ---")
    print(f" Best student accuracy: {best_acc:.2f}")
    teacher_acc, _, _ = validate(val_loader, teacher_model, criterion_ce, opt)
    print(f" Final teacher accuracy: {teacher_acc:.2f}")

    # Save the final model
    state = {'opt': opt, 'model': student_model.state_dict()}
    save_file = os.path.join(opt.save_folder, f'{opt.model_s}_last.pth')
    torch.save(state, save_file)
    print(f"Final model saved to {save_file}")






    #-------------ELABTORATING RESULTS-------------------------
    y_true = get_y_true(val_loader)
    y_teacher = get_y_teacher(teacher_model, val_loader)
    y_student = get_y_student(student_model, val_loader)
    
    print('==> ELABTORATING RESULTS...')
    analysis_results(teacher_model ,student_model, y_true, y_teacher, y_student)
    print('==> FINISH !')


if __name__ == '__main__':
    main()