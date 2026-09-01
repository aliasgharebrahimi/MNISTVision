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

<hr>

**Conclusion:**

Well, we need to determine the optimal batch size based on the fundamental rule: first, identify the batch size that yields the best model **accuracy**; then, **if other batch sizes** produce accuracy figures close to that peak, consider factors like **runtime** and **VRAM** usage.

**So, when determining the gap between two batch sizes—if that gap is small—how do we go about looking at runtime and VRAM?**

There is a simple general rule.

- **If the difference was less than 1%:** The remaining parameters must be seriously examined.
- **If the difference was between 1% and 3%:** You could also pay attention to the other parameters, but it depends heavily on VRAM and runtime.
- **If it was more than 3%:** Generally, a lower validation loss is preferable.

<hr>

**The best batch size?**

Well, the best accuracy here is for a batch size of **16**.

Now, regarding batch sizes close to 16, we can consider batch sizes of **2** and **4**, as the difference is minimal. However, if we look at the **runtime**, the runtimes for batch sizes 2 and 4 are significantly higher than that of batch size **16**; therefore, **16** is the best choice for this model's batch size.

<hr>

**Batch Size:** 16