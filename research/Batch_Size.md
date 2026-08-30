# Batch Size
> Well, welcome to the batch size experiment room.

Here, we share with you our **research** on selecting the appropriate **batch size**.
Here, we determine the appropriate batch size by tracking the experiment using the wandb tool.

<hr>

**wandb test results link:**

https://wandb.ai/aliasghare1-ai-emacv/MNISTVision/table?nw=nwuseraliasghare1ai

<hr>

**Batch Size Results Table:**

| Batch Size | LR    | Kernel Size | Optimizer | Layers | Epoch | Runtime | Vram | Train Loss | Val Loss |
|------------|-------|-------------|-----------|--------|-------|---------|------|------------|----------|
| 128        | 0.001 | 3           | Adam      | 2      | 2     | 29s     | -    | 0.17552    | 0.13261  |
| 64         | 0.001 | 3           | Adam      | 2      | 2     | 31s     | -    | 0.13255    | 0.10125  |
| 32         | 0.001 | 3           | Adam      | 2      | 2     | 36s     | -    | 0.10629    | 0.079299 |
| 16         | 0.001 | 3           | Adam      | 2      | 2     | 51s     | -    | 0.080036   | 0.056807 |
| 8          | 0.001 | 3           | Adam      | 2      | 2     | 1m 8s   | -    | 0.080036   | 0.060195 |
| 4          | 0.001 | 3           | Adam      | 2      | 2     | 1m 51s  | -    | 0.080619   | 0.058984 |
| 2          | 0.001 | 3           | Adam      | 2      | 2     | 3m 12s  | -    | 0.073757   | 0.058649 |
