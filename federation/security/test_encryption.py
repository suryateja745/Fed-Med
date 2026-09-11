"""
Unit test suite for FedMed Parameter Encryption, Differential Privacy, and Secure Aggregation.
Tests homomorphic encryption, additive masking, Gaussian differential privacy,
and arithmetic tolerance verification.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.security.encryption import (
    EncryptionContextManager,
    SecureClientHook,
    SecureServerHook,
    apply_differential_privacy,
    decrypt_parameters,
    encrypt_parameters,
    secure_aggregate_encrypted,
)


class TestSecurityEncryption(unittest.TestCase):
    """Test suite for parameter encryption, differential privacy, and secure aggregation."""

    def setUp(self):
        self.context_mgr = EncryptionContextManager(scheme="CKKS")
        self.dummy_params = [
            np.array([0.1, -0.25, 0.45, 0.8], dtype=np.float32),
            np.array([[0.05, 0.12], [-0.34, 0.56]], dtype=np.float32),
        ]

    def test_encrypt_and_decrypt_parameters(self):
        """Test that encryption followed by decryption recovers original weights within numerical tolerance."""
        encrypted = encrypt_parameters(self.dummy_params, context=self.context_mgr)
        decrypted = decrypt_parameters(encrypted, context=self.context_mgr)

        self.assertEqual(len(decrypted), len(self.dummy_params))
        for orig, dec in zip(self.dummy_params, decrypted):
            self.assertEqual(orig.shape, dec.shape)
            # Verify arithmetic tolerance (epsilon < 1e-4)
            np.testing.assert_allclose(orig, dec, rtol=1e-3, atol=1e-4)

    def test_secure_homomorphic_aggregation(self):
        """Test that secure aggregation in ciphertext domain accurately computes weighted average."""
        # Client 1 params (weight 2.0)
        p1 = [
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
        ]
        # Client 2 params (weight 1.0)
        p2 = [
            np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        ]

        # Expected weighted average: (2*p1 + 1*p2) / 3
        expected = [
            (2.0 * p1[0] + 1.0 * p2[0]) / 3.0,
            (2.0 * p1[1] + 1.0 * p2[1]) / 3.0,
        ]

        enc1 = encrypt_parameters(p1, context=self.context_mgr)
        enc2 = encrypt_parameters(p2, context=self.context_mgr)

        agg_enc = secure_aggregate_encrypted([(enc1, 2.0), (enc2, 1.0)])
        agg_dec = decrypt_parameters(agg_enc, context=self.context_mgr)

        for exp, act in zip(expected, agg_dec):
            np.testing.assert_allclose(exp, act, rtol=1e-3, atol=1e-4)

    def test_differential_privacy_clipping_and_noise(self):
        """Test Differential Privacy clipping and Gaussian noise addition."""
        large_params = [
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
            np.array([5.0, -15.0], dtype=np.float32),
        ]
        clip_norm = 2.0

        # Without noise (sigma = 0)
        clipped_params, telemetry = apply_differential_privacy(
            parameters=large_params,
            epsilon=0.0,
            delta=0.0,
            clip_norm=clip_norm,
        )

        total_norm = np.sqrt(sum(np.sum(np.square(arr)) for arr in clipped_params))
        self.assertAlmostEqual(total_norm, clip_norm, places=4)
        self.assertAlmostEqual(telemetry["clipped_l2_norm"], clip_norm, places=4)

        # With calibrated noise
        noisy_params, noisy_telemetry = apply_differential_privacy(
            parameters=large_params,
            epsilon=1.0,
            delta=1e-5,
            clip_norm=clip_norm,
            seed=42,
        )
        self.assertGreater(noisy_telemetry["noise_sigma"], 0.0)
        self.assertEqual(len(noisy_params), len(large_params))

    def test_client_server_security_hooks(self):
        """Test end-to-end SecureClientHook and SecureServerHook."""
        client_hook = SecureClientHook(
            enable_encryption=True,
            enable_dp=True,
            epsilon=2.0,
            clip_norm=5.0,
            context_manager=self.context_mgr,
        )
        server_hook = SecureServerHook(context_manager=self.context_mgr)

        outbound_payload, telemetry = client_hook.process_outbound_parameters(self.dummy_params)
        self.assertTrue(telemetry["encrypted"])
        self.assertTrue(telemetry["dp_applied"])

        # Server aggregates
        aggregated = server_hook.aggregate_and_decrypt([(outbound_payload, 1.0)])
        self.assertEqual(len(aggregated), len(self.dummy_params))
        for orig, res in zip(self.dummy_params, aggregated):
            self.assertEqual(orig.shape, res.shape)


if __name__ == "__main__":
    unittest.main()
