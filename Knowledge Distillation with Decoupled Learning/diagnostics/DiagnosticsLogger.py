import time
import csv
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class DiagnosticsLogger:
    def __init__(self, model, writer: SummaryWriter, modules_to_monitor=None, csv_path=None):
        """
        model: neural network whose parameters and activations will be monitored
        writer: TensorBoard SummaryWriter
        modules_to_monitor: list of nn.Module names (strings) to log activations
        csv_path: path to CSV file for exporting diagnostics (optional)
        """
        self.model = model
        self.writer = writer
        self.global_step = 0
        # Store previous weights for delta calculation
        self.prev_weights = {
            name: param.data.clone().detach().cpu()
            for name, param in model.named_parameters()
        }
        # Activation stats storage
        self.activation_buffer = {}
        # Optionally open CSV
        self.csv_file = None
        if csv_path:
            self.csv_file = open(csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            header = ["step", "param_name", "weight_norm", "grad_norm", "delta_weight_norm"]
            # activation columns added dynamically per hook
            self.csv_writer.writerow(header)

        # Register activation hooks
        if modules_to_monitor is None:
            modules_to_monitor = [name for name, m in model.named_modules() if isinstance(m, nn.ReLU)]
        for name, module in model.named_modules():
            if name in modules_to_monitor:
                module.register_forward_hook(self._get_activation_hook(name))

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            act = output.detach()
            # store for writing in log_step
            self.activation_buffer[name] = {
                'mean': act.mean().item(),
                'std': act.std().item()
            }
            # TensorBoard logging
            self.writer.add_scalar(f"activation/{name}_mean", self.activation_buffer[name]['mean'], self.global_step)
            self.writer.add_scalar(f"activation/{name}_std", self.activation_buffer[name]['std'], self.global_step)
        return hook

    def log_step(self, optimizer: torch.optim.Optimizer):
        """
        Call this every batch (or epoch) to log gradients, weights, deltas, and optionally export to CSV
        optimizer: optimizer used in training
        """
        self.global_step += 1
        # Log parameter norms and gradient norms
        for name, param in self.model.named_parameters():
            w = param.data
            g = param.grad
            w_norm = w.norm().item()
            self.writer.add_scalar(f"weight_norm/{name}", w_norm, self.global_step)
            g_norm = g.norm().item() if g is not None else 0.0
            self.writer.add_scalar(f"grad_norm/{name}", g_norm, self.global_step)
            prev_w = self.prev_weights[name]
            delta_norm = (w.cpu() - prev_w).norm().item()
            self.writer.add_scalar(f"delta_weight_norm/{name}", delta_norm, self.global_step)
            # CSV export
            if self.csv_file:
                self.csv_writer.writerow([self.global_step, name, w_norm, g_norm, delta_norm])
            self.prev_weights[name] = w.detach().cpu().clone()
        # CSV flush
        if self.csv_file:
            self.csv_file.flush()

    def close(self):
        """Close any open resources (CSV file)."""
        if self.csv_file:
            self.csv_file.close()
