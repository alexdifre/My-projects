import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analysis_results(teacher, STUDENT, y_true, y_teacher, y_student):
    
    TEACHER = get_class_name_from_type(type(teacher))
    
    # Setup del percorso per salvare le immagini
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    images_dir = os.path.join(parent_dir, "CONTR_DISTILL", "images")
    os.makedirs(images_dir, exist_ok=True)

    _, y_teacher_pred = y_teacher.max(dim=1)
    teacher_correct = (y_teacher_pred == y_true)

    _, y_student_pred = y_student.max(dim=1)
    student_correct = (y_student_pred == y_true)

    a = (teacher_correct & student_correct).sum().item()
    b = (teacher_correct & (~student_correct)).sum().item()
    c = ((~teacher_correct) & student_correct).sum().item()
    d = ((~teacher_correct) & (~student_correct)).sum().item()

    table = [[a, b], [c, d]]

    # 1) McNemar per performance
    res_mc = mcnemar(table, exact=False, correction=True)
    res_ex = mcnemar(table, exact=True)

    print("\n" + "="*60)
    print("McNemar Test - Model Performance Comparison")
    print("="*60)
    print(f"Contingency table:")
    print(f"  Both correct: {a}")
    print(f"  {TEACHER} correct / {STUDENT} wrong: {b}")
    print(f"  {TEACHER} wrong / {STUDENT} correct: {c}")
    print(f"  Both wrong: {d}\n")

    # -----------------Plotting chi-squared distribution--------------- 
    alpha = 0.05
    chi2_stat = res_mc.statistic
    p_value = res_mc.pvalue
    df = 1  # Degrees of freedom per McNemar test

    chi2_critical = chi2.ppf(1 - alpha, df=df)
    
    # -----------------Plotting chi-squared distribution--------------- 
    alpha = 0.05
    chi2_stat = res_mc.statistic
    p_value = res_mc.pvalue
    df = 1  # Degrees of freedom per McNemar test

    chi2_critical = chi2.ppf(1 - alpha, df=df)

    # ----------------- Plot Configuration -----------------
    # Create figure BEFORE plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Better x-axis range for df=1 (chi-square distribution is very skewed)
    # For df=1, most of the distribution is concentrated near 0
    x_max = max(15, chi2_stat * 1.2, chi2_critical * 1.5)
    x = np.linspace(0.001, x_max, 2000)  # Start from 0.001 to avoid division issues
    y = chi2.pdf(x, df)

    # Base plot with thicker line
    ax.plot(x, y, 'b-', lw=3, label=f'$\\chi^2$ distribution (df={df})')

    # Rejection region (right tail) - fill area
    if chi2_critical < x_max:
        x_reject = np.linspace(chi2_critical, x_max, 500)
        y_reject = chi2.pdf(x_reject, df)
        ax.fill_between(x_reject, y_reject, color='red', alpha=0.3, label=f'Rejection Region (α={alpha})')

    # Critical value line
    ax.axvline(chi2_critical, color='red', linestyle='--', lw=3, 
                label=f'Critical Value = {chi2_critical:.3f}')

    # Test statistic line (only if it's within reasonable range)
    if chi2_stat <= x_max:
        ax.axvline(chi2_stat, color='darkgreen', linestyle='-', lw=3, 
                    label=f'Test Statistic = {chi2_stat:.3f}')

    # P-value annotation with better positioning
    p_text = f'p-value = {p_value:.3e}' if p_value < 0.001 else f'p-value = {p_value:.4f}'

    # Decision text
    if p_value < alpha:
        decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
        decision_color = 'red' if p_value < alpha else 'green'
    else:
        decision = "FAIL TO REJECT H₀"
        decision_color = 'green'


    # Results text box
    p_text = f'p-value = {p_value:.3e}' if p_value < 0.001 else f'p-value = {p_value:.4f}'
    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"

    textstr = f'{p_text}\n{decision}\n(α = {alpha})'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.65, 0.85, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')

    # Formatting with better labels
    ax.set_title(f'McNemar Test: {TEACHER} vs {STUDENT}\nChi-Square Distribution (df = {df})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Chi-Square Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits for df=1
    ax.set_xlim(0, 10)  
    ax.set_ylim(0, 1)  

    # Improve layout
    plt.tight_layout()
    
    # Labels and formatting
    ax.set_title(f'McNemar Test: {TEACHER} vs {STUDENT}\nChi-Square Distribution (df = {df})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Chi-Square Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nMcNemar Test Results: {TEACHER} vs {STUDENT}")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.5f}")
    print(f"Critical value (α=0.05): {chi2_critical:.3f}")
    print(f"Decision: {decision}")

    if res_mc.pvalue < 0.05:
        print("\nPERFORMANCE CONCLUSION: Significant difference (p < 0.05)")
        if b > c:
            print(f"→ {TEACHER} significantly better than {STUDENT}")
        else:
            print(f"→ {STUDENT} significantly better than {TEACHER}")
    else:
        print(f"\nPERFORMANCE CONCLUSION: No significant difference between {STUDENT} and {TEACHER}")
        
    # Test d'indipendenza degli errori
    print("\n" + "="*60)
    print("Error Independence Test")
    print("="*60)
    
    # Calcolo del test chi-quadro di indipendenza
    chi2_stat_indep, pval, _, _ = chi2_contingency(table)
    
    # Calcolo del coefficiente Phi
    n = a + b + c + d  # Totale campioni
    phi = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    
    # ----------------CONTINGENCY TABLE HEATMAP---------
    plt.figure(figsize=(8, 6))
    sns.heatmap([[table[0][0], table[0][1]], 
                [table[1][0], table[1][1]]], 
                annot=True, fmt="d", cmap="YlOrRd", 
                xticklabels=[f"{STUDENT} Correct", f"{STUDENT} Wrong"],
                yticklabels=[f"{TEACHER} Correct", f"{TEACHER} Wrong"])
    plt.title(f"Contingency Table: {TEACHER} vs {STUDENT}\nError Independence Analysis (Phi = {phi:.3f})")
    
    # Salva la heatmap della tabella di contingenza
    heatmap_filename = f"{TEACHER}_vs_{STUDENT}_contingency_heatmap.png"
    heatmap_filepath = os.path.join(images_dir, heatmap_filename)
    plt.savefig(heatmap_filepath, dpi=300, bbox_inches='tight')
    print(f"🔥 Saved contingency table heatmap: {heatmap_filename}")
    plt.show()
    
    print(f"Chi-square statistic: {chi2_stat_indep:.4f}")
    print(f"p-value: {pval:.5f}")
    print(f"Phi coefficient (ϕ): {phi:.3f}")

    # Interpretazione
    if pval < 0.05:
        print("\nINDEPENDENCE CONCLUSION: Errors are NOT independent (p < 0.05)")
        if phi > 0.3:
            print("→ STRONG POSITIVE CORRELATION (ϕ > 0.3)")
            print("   Models make mistakes on the same samples")
        elif phi < -0.3:
            print("→ STRONG NEGATIVE CORRELATION (ϕ < -0.3)")
            print("   Models make mistakes on different samples (complementary)")
        elif phi > 0:
            print("→ Positive correlation (0 < ϕ ≤ 0.3)")
            print("   Models tend to make mistakes on the same samples")
        elif phi < 0:
            print("→ Negative correlation (-0.3 ≤ ϕ < 0)")
            print("   Models tend to make mistakes on different samples")
        else:
            print("→ No correlation (ϕ ≈ 0)")
    else:
        print("\nINDEPENDENCE CONCLUSION: Errors are statistically independent (p ≥ 0.05)")

    # PASSAGGIO 6: Calcolo dell'indice di separazione
    overlap = ((~teacher_correct) & (~student_correct)).sum().item()
    total_errors = (~teacher_correct).sum().item() + (~student_correct).sum().item()

    separation_index = 1 - (2 * overlap / total_errors) if total_errors > 0 else 0.0

    print("\n" + "="*60)
    print("Additional Metrics")
    print("="*60)
    print(f"Accuracy {TEACHER}: {teacher_correct.float().mean().item() * 100:.2f}%")
    print(f"Accuracy {STUDENT}: {student_correct.float().mean().item() * 100:.2f}%")
    print(f"Error Separation Index: {separation_index:.2f} [0=same errors, 1=completely different errors]")

    # ----------------ACCURACY COMPARISON BAR PLOT---------
    # Calculate accuracies
    teacher_accuracy = teacher_correct.float().mean().item() * 100
    student_accuracy = student_correct.float().mean().item() * 100
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    models = [TEACHER, STUDENT]
    accuracies = [teacher_accuracy, student_accuracy]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    plt.title(f'Accuracy Comparison: {TEACHER} vs {STUDENT}', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.ylim(0, 100)

    # Aggiungi valori sulle barre
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    
    # Salva il grafico delle accuracy
    accuracy_filename = f"{TEACHER}_vs_{STUDENT}_accuracy_comparison.png"
    accuracy_filepath = os.path.join(images_dir, accuracy_filename)
    plt.savefig(accuracy_filepath, dpi=300, bbox_inches='tight')
    print(f"📈 Saved accuracy comparison plot: {accuracy_filename}")
    plt.show()

    print(f"\n🎯 All images saved in: {images_dir}")


def get_class_name(obj) -> str:
    return obj.__class__.__name__

def get_class_name_from_type(cls) -> str:
    if isinstance(cls, type):
        return cls.__name__
    else:
        raise TypeError("Input must be a class type, not an instance.")