# Activation Function

## Review of the Theoretical Aspects

For the initial implementation, I chose **ReLU (Rectified Linear Unit)** as the activation function. The main reason was its ability to reduce the impact of the **vanishing gradient problem**, while also keeping the network computationally efficient and relatively lightweight.

However, after further research, I found that ReLU can suffer from the **"Leaky ReLU" problem**, where neurons may become inactive and stop contributing to the learning process.

To address this limitation, I replaced ReLU with **Leaky ReLU**. By allowing a small gradient to pass through for negative input values, Leaky ReLU provides a practical alternative that helps prevent neurons from becoming permanently inactive.

Based on these considerations, **Leaky ReLU was selected as the activation function for the current implementation.**