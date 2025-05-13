#!/usr/bin/env python
import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the src directory to the path so we can import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer.training.util import rate

def plot_learning_rate_schedule():
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Original implementation parameters
    original_model_size = 512
    original_factor = 1.0
    original_warmup = 4000
    original_steps = 100000
    
    # Current implementation parameters
    current_model_size = 512
    current_factor = 1.0
    current_warmup = 3000
    current_steps = 20000
    
    # Function to simulate and collect learning rates
    def get_learning_rates(model_size, factor, warmup, steps):
        dummy_model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        
        lr_scheduler = LambdaLR(
            optimizer=optimizer, 
            lr_lambda=lambda step: rate(step, model_size, factor, warmup)
        )
        
        learning_rates = []
        for step in range(steps):
            learning_rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        
        return np.array(learning_rates)
    
    # Get learning rates for both implementations
    original_lr = get_learning_rates(original_model_size, original_factor, original_warmup, original_steps)
    current_lr = get_learning_rates(current_model_size, current_factor, current_warmup, current_steps)
    
    # Plot current implementation
    plt.subplot(2, 1, 1)
    plt.plot(range(current_steps), current_lr, label=f'Current: model_size={current_model_size}, warmup={current_warmup}')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Current Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line at warmup step
    plt.axvline(x=current_warmup, color='r', linestyle='--', alpha=0.5, label='Warmup Completed')
    plt.axvline(x=2200, color='g', linestyle='--', alpha=0.5, label='[BS=32] Epoch 100')
    plt.axvline(x=8900, color='b', linestyle='--', alpha=0.5, label='[BS=128] Epoch 50')
    
    # Zoom in on the important part of the curve
    plt.xlim([0, 10000])
    plt.legend()
    
    # Plot original implementation
    plt.subplot(2, 1, 2)
    plt.plot(range(original_steps), original_lr, 'orange', label=f'Original: model_size={original_model_size}, warmup={original_warmup}')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Original Implementation Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line at warmup step
    plt.axvline(x=original_warmup, color='r', linestyle='--', alpha=0.5, label='Warmup Completed')
    
    # Show first 20000 steps for better comparison
    plt.xlim([0, 20000])
    plt.legend()

    plt.tight_layout()
    
    # Save the plots
    plt.savefig('learning_rate_schedule_comparison.png')
    print(f"Learning rate schedule comparison plot saved as 'learning_rate_schedule_comparison.png'")
    
    # Also save the original plots to maintain compatibility
    plt.figure(figsize=(10, 6))
    plt.plot(range(current_steps), current_lr, label=f'model_size={current_model_size}, warmup={current_warmup}')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line at warmup step
    plt.axvline(x=current_warmup, color='r', linestyle='--', alpha=0.5, label='Warmup Completed')
    
    # Zoom in on the important part of the curve
    plt.xlim([0, 10000])
    
    plt.legend()

    # Save the plot
    plt.savefig('learning_rate_schedule.png')
    print(f"Learning rate schedule plot saved as 'learning_rate_schedule.png'")
    
    plt.axvline(x=2200, color='g', linestyle='--', alpha=0.5, label='[BS=32] Epoch 100')
    plt.axvline(x=8900, color='b', linestyle='--', alpha=0.5, label='[BS=128] Epoch 50')
    plt.legend()
    plt.savefig('learning_rate_schedule_with_epochs.png')
    
    # Generate a full view of original schedule
    plt.figure(figsize=(10, 6))
    plt.plot(range(original_steps), original_lr, 'orange', label=f'Original: model_size={original_model_size}, warmup={original_warmup}')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Full Original Implementation Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(x=original_warmup, color='r', linestyle='--', alpha=0.5, label='Warmup Completed')
    plt.legend()
    plt.savefig('original_learning_rate_schedule.png')
    print(f"Original learning rate schedule plot saved as 'original_learning_rate_schedule.png'")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    plot_learning_rate_schedule() 