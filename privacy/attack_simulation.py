import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__))
from differential_privacy import add_differential_privacy_noise

def simulate_inversion_attack(true_weight, noisy_weight):
    """
    A naive 'attacker' assumes the noisy weight IS the true weight.
    We measure how wrong that guess is.
    """
    error = abs(true_weight - noisy_weight)
    return error

if __name__ == "__main__":
    true_weights = [0.42, 0.17, 0.88]

    print("=== WITHOUT Differential Privacy (no protection) ===")
    for w in true_weights:
        print(f"True weight: {w} | Attacker sees: {w} | Attack error: 0.0 (fully exposed!)")

    print("\n=== WITH Differential Privacy (epsilon=0.5, strong protection) ===")
    noisy = add_differential_privacy_noise(true_weights, epsilon=0.5)
    total_error = 0
    for true_w, noisy_w in zip(true_weights, noisy):
        error = simulate_inversion_attack(true_w, noisy_w)
        total_error += error
        print(f"True weight: {true_w:.4f} | Attacker sees: {noisy_w:.4f} | Attack error: {error:.4f}")

    avg_error = total_error / len(true_weights)
    print(f"\nAverage attack error with DP protection: {avg_error:.4f}")
    print("Conclusion: The attacker cannot reliably recover the true weight — privacy preserved.")