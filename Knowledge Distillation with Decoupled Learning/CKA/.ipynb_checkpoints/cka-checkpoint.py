import torch

def cka(f_s, f_t):
    """
    Calcola il Centered Kernel Alignment tra due set di feature.
    nel nostro caso lo eseguiremo solo nell ultimo layer poichè
    nonostante abbiamo allenato parzialmente quello intermedio è 
    poco informativo, lo è di più quello dell ultimo e non ci aspettiamo e
    ,sopratutto, non vogliamo raggiungere livelli troppo vicini a 1. 
    
    Args:
        f_s: Tensor di shape (n_samples, n_features1)
        f_t: Tensor di shape (n_samples, n_features2)
        
    output:
        cka (float):tra 0 e 1.
    """
    
    n = f_s.shape[0]
    
    # 1. Centratura delle feature
    features1_centered = f_s - f_s.mean(dim=0, keepdim=True)
    features2_centered = f_t - f_t.mean(dim=0, keepdim=True)
    
    # 2. Calcolo dei kernel (Gram matrices)
    kernel1 = torch.mm(features1_centered, features1_centered.T)  # shape (n, n)
    kernel2 = torch.mm(features2_centered, features2_centered.T)  # shape (n, n)
    
    # 3. Centratura dei kernel
    # Sottrai la media delle righe e delle colonne
    row_means = kernel1.mean(dim=1, keepdim=True)
    col_means = kernel1.mean(dim=0, keepdim=True)
    overall_mean = kernel1.mean()
    kernel1_centered = kernel1 - row_means - col_means + overall_mean
    
    row_means = kernel2.mean(dim=1, keepdim=True)
    col_means = kernel2.mean(dim=0, keepdim=True)
    overall_mean = kernel2.mean()
    kernel2_centered = kernel2 - row_means - col_means + overall_mean
    
    # 4. Calcolo del CKA
    hsic = torch.norm(torch.mm(kernel1_centered, kernel2_centered)) ** 2
    normalization = torch.norm(kernel1_centered) ** 2 * torch.norm(kernel2_centered) ** 2
    
    cka = hsic / normalization
    return cka.item()
