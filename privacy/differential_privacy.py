import numpy as np

def add_differential_privacy_noise(weights, epsilon=1.0, sensitivity=1.0):
    """
    Adds Laplace noise to weights to protect against model inversion attacks.
    epsilon: privacy budget - smaller = more private (more noise), larger = less private (less noise)
    sensitivity: how much a single data point can affect the result
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=len(weights))
    noisy_weights = [w + n for w, n in zip(weights, noise)]
    return noisy_weights

if __name__ == "__main__":
    original_weights = [0.10, 0.20, 0.30, 0.40]

    print(f"Original weights: {original_weights}")

    noisy_strong = add_differential_privacy_noise(original_weights, epsilon=0.1)
    print(f"With STRONG privacy (epsilon=0.1): {noisy_strong}")

    noisy_weak = add_differential_privacy_noise(original_weights, epsilon=10.0)
    print(f"With WEAK privacy (epsilon=10.0): {noisy_weak}")