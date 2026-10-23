# Layer Output Shapes
> In this test room, we examine the output dimensions of each layer so that we can correctly configure the layer inputs without triggering errors.

---

## Calculating outputs

Well, we use the following formula for the calculation:

$$
O = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1
$$

Where:

- $N$: Input size
- $F$: Filter (kernel) size
- $P$: Padding amount
- $S$: Stride value
- $O$: Output size

We can also obtain this using this formula and the 'summary' function; in fact, that is how we do it—by using 'summary'.

```python
from torchinfo import summary

summary(model, input_size=(1, 1, 28, 28))
```

Use this code to obtain the output of each layer and adjust the values.

Code output:

| Layer (type:depth-idx) | Output Shape | Param # |
|---|---|---:|
| MNISTNet | `[1, 10]` | -- |
| ├─ ConvBlock: 1-1 | `[1, 6, 13, 13]` | -- |
| │ ├─ Conv2d: 2-1 | `[1, 6, 26, 26]` | 60 |
| │ ├─ BatchNorm2d: 2-2 | `[1, 6, 26, 26]` | 12 |
| │ ├─ LeakyReLU: 2-3 | `[1, 6, 26, 26]` | -- |
| │ └─ MaxPool2d: 2-4 | `[1, 6, 13, 13]` | -- |
| ├─ ConvBlock: 1-2 | `[1, 12, 5, 5]` | -- |
| │ ├─ Conv2d: 2-5 | `[1, 12, 11, 11]` | 660 |
| │ ├─ BatchNorm2d: 2-6 | `[1, 12, 11, 11]` | 24 |
| │ ├─ LeakyReLU: 2-7 | `[1, 12, 11, 11]` | -- |
| │ └─ MaxPool2d: 2-8 | `[1, 12, 5, 5]` | -- |
| ├─ Flatten: 1-3 | `[1, 300]` | -- |
| └─ Linear: 1-4 | `[1, 10]` | 3,010 |