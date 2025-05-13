# GPU=4 Performance Experiment Results

## Summary Table

| Batch Size | Accum. Steps | Total Exec. Time | Train Worker | Epoch | Training | Validation | Backward Pass | Loss Comp. | Forward Pass | Tokens/Sec |
|------------|--------------|------------------|--------------|-------|----------|------------|---------------|------------|--------------|------------|
| 16         | 64           | 314.00s          | 300.29s      | 268.04s | 253.84s  | 12.97s     | 111.54s (0.0314s) | 2.12s (0.0005s) | 72.86s (0.0182s) | ~2,090 |
| 32         | 32           | 184.91s          | 171.14s      | 138.92s | 130.85s  | 6.85s      | 55.51s (0.0313s) | 1.53s (0.0008s) | 37.37s (0.0187s) | ~4,040 |
| 64         | 16           | 118.89s          | 105.70s      | 73.79s  | 68.95s   | 3.62s      | 28.68s (0.0323s) | 1.19s (0.0012s) | 18.15s (0.0182s) | ~7,760 |
| 128        | 8            | 95.20s           | 82.20s       | 50.68s  | 47.22s   | 2.19s      | 15.31s (0.0345s) | 3.09s (0.0062s) | 9.40s (0.0188s) | ~11,360 |
| 256        | 4            | 89.86s           | 76.43s       | 42.13s  | 38.96s   | 1.97s      | 13.10s (0.0590s) | 5.86s (0.0234s) | 4.87s (0.0195s) | ~13,900 |
| 512        | 2            | 86.43s           | 73.23s       | 38.74s  | 35.54s   | 1.91s      | 12.13s (0.1093s) | 7.38s (0.0590s) | 2.68s (0.0214s) | ~15,250 |
| 1024       | 1            | 102.82s          | 88.91s       | 36.43s  | 33.33s   | 1.78s      | 10.97s (0.1994s) | 7.89s (0.1273s) | 1.59s (0.0257s) | ~16,150 |

## Key Observations

1. **Execution Time**: Performance improves steadily with increasing batch size up to BS=512, but degrades slightly at BS=1024, likely due to increased overhead or memory limitations.

2. **Throughput**: Tokens processed per second increases consistently with batch size, from ~2,090 at BS=16 to ~16,150 at BS=1024, showing strong scaling.

3. **Training Efficiency**: Training time decreases significantly from BS=16 (253.84s) to BS=1024 (33.33s), a ~7.6x improvement.

4. **Backward Pass**: Average backward pass time per step increases with batch size (0.0314s at BS=16 to 0.1994s at BS=1024), as expected due to larger gradient computations.

5. **Validation**: Validation time decreases with batch size, from 12.97s at BS=16 to 1.78s at BS=1024, showing efficient parallel processing.

6. **Initialization Overhead**: Initialization times are relatively consistent across batch sizes, but become a higher percentage of total execution time at larger batch sizes.

7. **Optimal Configuration**: For this specific model and dataset on 4 GPUs, batch size 512 with 2 accumulation steps appears to be the most efficient configuration, with the lowest total execution time.

## Timing Details by Batch Size

### Batch Size = 16, Accumulation Steps = 64
- Total script execution time: 314.00 seconds
- Train worker: 300.29s
- Train worker.epoch: 268.04s
- Train worker.epoch.training: 253.84s
- Train worker.epoch.validation: 12.97s
- Backward pass: 111.54s (mean=0.0314s)
- Loss computation: 2.12s (mean=0.0005s)
- Forward pass: 72.86s (mean=0.0182s)

### Batch Size = 32, Accumulation Steps = 32
- Total script execution time: 184.91 seconds
- Train worker: 171.14s
- Train worker.epoch: 138.92s
- Train worker.epoch.training: 130.85s
- Train worker.epoch.validation: 6.85s
- Backward pass: 55.51s (mean=0.0313s)
- Loss computation: 1.53s (mean=0.0008s)
- Forward pass: 37.37s (mean=0.0187s)

### Batch Size = 64, Accumulation Steps = 16
- Total script execution time: 118.89 seconds
- Train worker: 105.70s
- Train worker.epoch: 73.79s
- Train worker.epoch.training: 68.95s
- Train worker.epoch.validation: 3.62s
- Backward pass: 28.68s (mean=0.0323s)
- Loss computation: 1.19s (mean=0.0012s)
- Forward pass: 18.15s (mean=0.0182s)

### Batch Size = 128, Accumulation Steps = 8
- Total script execution time: 95.20 seconds
- Train worker: 82.20s
- Train worker.epoch: 50.68s
- Train worker.epoch.training: 47.22s
- Train worker.epoch.validation: 2.19s
- Backward pass: 15.31s (mean=0.0345s)
- Loss computation: 3.09s (mean=0.0062s)
- Forward pass: 9.40s (mean=0.0188s)

### Batch Size = 256, Accumulation Steps = 4
- Total script execution time: 89.86 seconds
- Train worker: 76.43s
- Train worker.epoch: 42.13s
- Train worker.epoch.training: 38.96s
- Train worker.epoch.validation: 1.97s
- Backward pass: 13.10s (mean=0.0590s)
- Loss computation: 5.86s (mean=0.0234s)
- Forward pass: 4.87s (mean=0.0195s)

### Batch Size = 512, Accumulation Steps = 2
- Total script execution time: 86.43 seconds
- Train worker: 73.23s
- Train worker.epoch: 38.74s
- Train worker.epoch.training: 35.54s
- Train worker.epoch.validation: 1.91s
- Backward pass: 12.13s (mean=0.1093s)
- Loss computation: 7.38s (mean=0.0590s)
- Forward pass: 2.68s (mean=0.0214s)

### Batch Size = 1024, Accumulation Steps = 1
- Total script execution time: 102.82 seconds
- Train worker: 88.91s
- Train worker.epoch: 36.43s
- Train worker.epoch.training: 33.33s
- Train worker.epoch.validation: 1.78s
- Backward pass: 10.97s (mean=0.1994s)
- Loss computation: 7.89s (mean=0.1273s)
- Forward pass: 1.59s (mean=0.0257s) 