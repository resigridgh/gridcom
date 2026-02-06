
import torch

def binary_cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
            Binary cross-entropy loss computed manually (for sigmoid outputs).
                """
                    epsilon = 1e-7  # prevent log(0)
                        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
                            loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
                                return loss


                            def binary_classification(d: int, n: int, epochs: int = 10000, eta: float = 0.001):
                                    """
                                        Binary Classification with Linear and Nonlinear Layers
                                            """
                                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                                    # Data
                                                        X = torch.randn(n, d, dtype=torch.float32, device=device)
                                                            Y = (X.sum(dim=1, keepdim=True) > 2).float()

                                                                # Weights (scaled random init)
                                                                    current_dtype = torch.float32
                                                                        W1 = (torch.randn(d, 48, device=device, dtype=current_dtype)
                                                                                          * torch.sqrt(torch.tensor(1.0 / d, device=device, dtype=current_dtype))).requires_grad_(True)

                                                                            W2 = (torch.randn(48, 16, device=device, dtype=current_dtype)
                                                                                              * torch.sqrt(torch.tensor(1.0 / 48, device=device, dtype=current_dtype))).requires_grad_(True)

                                                                                W3 = (torch.randn(16, 32, device=device, dtype=current_dtype)
                                                                                                  * torch.sqrt(torch.tensor(1.0 / 16, device=device, dtype=current_dtype))).requires_grad_(True)

                                                                                    W4 = (torch.randn(32, 1, device=device, dtype=current_dtype)
                                                                                                      * torch.sqrt(torch.tensor(1.0 / 32, device=device, dtype=current_dtype))).requires_grad_(True)

                                                                                        train_losses = torch.zeros(epochs, device=device)

                                                                                            for epoch in range(epochs):
                                                                                                        # Forward pass
                                                                                                                Z1 = X @ W1
                                                                                                                        Z1 = Z1 @ W2
                                                                                                                                A1 = 1.0 / (1.0 + torch.exp(-Z1))  # sigmoid

                                                                                                                                        Z2 = A1 @ W3
                                                                                                                                                Z2 = Z2 @ W4
                                                                                                                                                        YPred = 1.0 / (1.0 + torch.exp(-Z2))  # sigmoid

                                                                                                                                                                # Loss
                                                                                                                                                                        train_loss = binary_cross_entropy_loss(YPred, Y)

                                                                                                                                                                                # Backward pass
                                                                                                                                                                                        train_loss.backward()

                                                                                                                                                                                                # Gradient descent update
                                                                                                                                                                                                        with torch.no_grad():
                                                                                                                                                                                                                        W1 -= eta * W1.grad
                                                                                                                                                                                                                                    W2 -= eta * W2.grad
                                                                                                                                                                                                                                                W3 -= eta * W3.grad
                                                                                                                                                                                                                                                            W4 -= eta * W4.grad

                                                                                                                                                                                                                                                                        # Zero gradients
                                                                                                                                                                                                                                                                                    W1.grad.zero_()
                                                                                                                                                                                                                                                                                                W2.grad.zero_()
                                                                                                                                                                                                                                                                                                            W3.grad.zero_()
                                                                                                                                                                                                                                                                                                                        W4.grad.zero_()

                                                                                                                                                                                                                                                                                                                                    train_losses[epoch] = train_loss

                                                                                                                                                                                                                                                                                                                                            if epoch % 100 == 0:
                                                                                                                                                                                                                                                                                                                                                            print(f"Epoch {epoch} loss: {train_loss.item():.4f}")

                                                                                                                                                                                                                                                                                                                                                                return [train_losses, W1, W2, W3, W4]

