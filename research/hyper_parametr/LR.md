# LR
> Welcome to the LR Regulation Research Room.

In this file, we aim to find the best LR for our project using the experiment tracking technique.

<hr>

**wandb test results link:**

https://wandb.ai/aliasghare1-ai-emacv/MNISTVision/table?nw=nwuseraliasghare1ai
 
**LR Size Results Table:**

| Batch Size | LR     | Kernel Size | Optimizer | Layers | Epoch | Runtime | Vram | Train Loss | Val Loss |
|------------|--------|-------------|-----------|--------|-------|---------|------|------------|----------|
| 16         | 0.0001 | 3           | Adam      | 2      | 2     | 29s     | -    | 0.17552    | 0.13261  |
| 16         | 0.0003 | 3           | Adam      | 2      | 2     | 31s     | -    | 0.13255    | 0.10125  |
| 16         | 0.0005 | 3           | Adam      | 2      | 2     | 36s     | -    | 0.10629    | 0.079299 |
| 16         | 0.001  | 3           | Adam      | 2      | 2     | 51s     | -    | 0.080036   | 0.056807 |
| 16         | 0.003  | 3           | Adam      | 2      | 2     | 1m 8s   | -    | 0.080036   | 0.060195 |
| 16         | 0.005  | 3           | Adam      | 2      | 2     | 1m 51s  | -    | 0.080619   | 0.058984 |
| 16         | 0.01   | 3           | Adam      | 2      | 2     | 3m 12s  | -    | 0.073757   | 0.058649 |

<hr>