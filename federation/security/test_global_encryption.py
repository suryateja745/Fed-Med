"""
Unit test suite for Global Model Weight Encryption & Key Management.
Tests AES-256-GCM authenticated encryption, HMAC-SHA256 integrity verification,
PBKDF2 key derivation, key rotation, tamper detection, and PyTorch state dict serialization.
"""

import hashlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.security.global_encryption import (
    CryptographicIntegrityError,
    EncryptionMetadata,
    GlobalKeyManager,
    GlobalWeightEncryptionManager,
    MAGIC_BYTES,
)


class TestGlobalWeightEncryption(unittest.TestCase):
    """Test suite verifying cryptographic correctness and zero-trust security."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.key_path = Path(self.temp_dir) / "test_global.key"
        self.key_manager = GlobalKeyManager(key_path=self.key_path, auto_generate=True)
        self.manager = GlobalWeightEncryptionManager(key_manager=self.key_manager)

        # Create sample PyTorch state dict
        self.dummy_state_dict = {
            "conv1.weight": torch.randn(8, 4, 3, 3, 3, dtype=torch.float32),
            "conv1.bias": torch.randn(8, dtype=torch.float32),
            "bn.running_mean": torch.zeros(8, dtype=torch.float32),
        }

        # Create sample NumPy parameters
        self.dummy_params = [
            np.random.randn(8, 4, 3, 3, 3).astype(np.float32),
            np.random.randn(8).astype(np.float32),
        ]

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_key_manager_generation_and_persistence(self):
        """Test symmetric key generation, fingerprint, and file storage."""
        self.assertEqual(len(self.key_manager.key), 32)
        fingerprint = self.key_manager.key_id
        self.assertEqual(len(fingerprint), 8)
        self.assertTrue(self.key_path.exists())

        # Load key in new manager instance
        loaded_km = GlobalKeyManager(key_path=self.key_path, auto_generate=False)
        self.assertEqual(loaded_km.key, self.key_manager.key)
        self.assertEqual(loaded_km.key_id, fingerprint)

    def test_pbkdf2_passphrase_derivation(self):
        """Test deterministic key derivation from passphrases."""
        km1 = GlobalKeyManager(passphrase="SecretHospitalPassphrase123!", auto_generate=False)
        km2 = GlobalKeyManager(passphrase="SecretHospitalPassphrase123!", auto_generate=False)
        self.assertEqual(km1.key, km2.key)
        self.assertEqual(km1.key_id, km2.key_id)

    def test_key_rotation(self):
        """Test that rotating key updates fingerprint and changes key value."""
        old_key = self.key_manager.key
        old_id = self.key_manager.key_id

        new_key, new_id = self.key_manager.rotate_key()
        self.assertNotEqual(old_key, new_key)
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(self.key_manager.key_id, new_id)

    def test_state_dict_encryption_and_decryption(self):
        """Test round-trip PyTorch state dict encryption, authentication, and recovery."""
        bundle_bytes = self.manager.encrypt_state_dict(
            state_dict=self.dummy_state_dict,
            round_num=3,
            dice_score=0.885,
        )
        self.assertTrue(bundle_bytes.startswith(MAGIC_BYTES))

        recovered_sd, metadata = self.manager.decrypt_state_dict(bundle_bytes)
        self.assertEqual(metadata.round_num, 3)
        self.assertAlmostEqual(metadata.dice_score, 0.885, places=3)
        self.assertEqual(metadata.cipher, "AES-256-GCM")
        self.assertEqual(metadata.key_id, self.manager.key_id)

        # Verify exact weight tensor equality
        for k in self.dummy_state_dict:
            self.assertIn(k, recovered_sd)
            torch.testing.assert_close(self.dummy_state_dict[k], recovered_sd[k])

    def test_numpy_parameters_encryption_and_decryption(self):
        """Test round-trip NumPy parameters encryption and recovery."""
        bundle_bytes = self.manager.encrypt_parameters(
            parameters=self.dummy_params,
            round_num=5,
            dice_score=0.912,
        )

        recovered_params, metadata = self.manager.decrypt_parameters(bundle_bytes)
        self.assertEqual(metadata.round_num, 5)
        self.assertEqual(len(recovered_params), len(self.dummy_params))
        for orig, rec in zip(self.dummy_params, recovered_params):
            np.testing.assert_array_equal(orig, rec)

    def test_checkpoint_file_encryption_and_decryption(self):
        """Test encrypting and decrypting a .pth checkpoint file on disk."""
        pth_path = Path(self.temp_dir) / "test_checkpoint.pth"
        torch.save(self.dummy_state_dict, pth_path)

        # Encrypt
        enc_path = self.manager.encrypt_checkpoint_file(
            source_path=pth_path,
            round_num=2,
            dice_score=0.84,
        )
        self.assertTrue(enc_path.exists())
        self.assertTrue(str(enc_path).endswith(".pth.enc"))

        # Decrypt
        dec_path, metadata = self.manager.decrypt_checkpoint_file(enc_path)
        self.assertTrue(dec_path.exists())
        self.assertEqual(metadata.round_num, 2)

        recovered = torch.load(dec_path, weights_only=False)
        torch.testing.assert_close(self.dummy_state_dict["conv1.weight"], recovered["conv1.weight"])

    def test_tamper_detection_and_integrity_rejection(self):
        """Test that altering a single byte in the ciphertext raises CryptographicIntegrityError."""
        bundle_bytes = bytearray(self.manager.encrypt_state_dict(self.dummy_state_dict))

        # Flip a bit in the ciphertext payload (middle of the bundle)
        mid_idx = len(bundle_bytes) // 2
        bundle_bytes[mid_idx] ^= 0x01

        # Attempt to decrypt corrupted bundle
        with self.assertRaises(CryptographicIntegrityError):
            self.manager.decrypt_state_dict(bytes(bundle_bytes))

    def test_bundle_verification_tool(self):
        """Test standalone integrity inspection tool."""
        bundle_bytes = self.manager.encrypt_state_dict(
            state_dict=self.dummy_state_dict,
            round_num=4,
            dice_score=0.87,
        )

        # Valid bundle verification
        res_valid = self.manager.verify_bundle_integrity(bundle_bytes)
        self.assertTrue(res_valid["valid"])
        self.assertEqual(res_valid["status"], "VERIFIED")
        self.assertEqual(res_valid["round_num"], 4)
        self.assertAlmostEqual(res_valid["dice_score"], 0.87, places=2)

        # Corrupted bundle verification
        corrupted = bytearray(bundle_bytes)
        corrupted[-10] ^= 0xFF
        res_invalid = self.manager.verify_bundle_integrity(bytes(corrupted))
        self.assertFalse(res_invalid["valid"])
        self.assertIn("MISMATCH", res_invalid["status"] + res_invalid.get("error", ""))


if __name__ == "__main__":
    unittest.main()
