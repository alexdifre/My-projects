import subprocess

student_models = ['resnet8', 'resnet14', 'resnet20']

for model in student_models:
    print(f"\nRunning distillation for student model: {model}\n")
    subprocess.run([
        "python3", "main.py",
        "--model_s", model
    ])
