import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the performance study
batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
exec_time = [314.00, 184.91, 118.89, 95.20, 89.86, 86.43, 102.82]
training_time = [253.84, 130.85, 68.95, 47.22, 38.96, 35.54, 33.33]
tokens_per_sec = [2090, 4040, 7760, 11360, 13900, 15250, 16150]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot execution and training time
ax1.plot(batch_sizes, exec_time, 'o-', color='#3498db', linewidth=2, label='Total Execution Time')
ax1.plot(batch_sizes, training_time, 'o-', color='#e74c3c', linewidth=2, label='Training Time')
ax1.set_xscale('log')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Execution vs Training Time by Batch Size')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels(batch_sizes)

# Plot tokens per second
ax2.plot(batch_sizes, tokens_per_sec, 'o-', color='#2ecc71', linewidth=2)
ax2.set_xlabel('Batch Size')
ax2.set_xscale('log')
ax2.set_ylabel('Tokens per Second')
ax2.set_title('Throughput by Batch Size')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(batch_sizes)
ax2.set_xticklabels(batch_sizes)

# Highlight optimal batch size
ax1.axvline(x=512, color='#9b59b6', linestyle='--', alpha=0.7, label='Optimal Batch Size')
ax2.axvline(x=512, color='#9b59b6', linestyle='--', alpha=0.7)

# Add annotations
ax1.annotate('Optimal: BS=512', 
            xy=(512, 86.43), 
            xytext=(600, 120),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=9)

ax2.annotate('Maximum throughput: BS=1024', 
            xy=(1024, 16150), 
            xytext=(700, 16300),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=9)

# Adjust layout
plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300)
plt.close()

print("Performance analysis graph saved as 'performance_analysis.png'") 