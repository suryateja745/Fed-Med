"""
Master Verification & Test Suite Runner for FedMed.
Runs all 14 unit test suites, verifies dataset partitioners, MONAI 3D U-Net models,
client-server communication, aggregation strategies, security encryption, triggers,
simulation harness, API bridge, and end-to-end dry-run workflows.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
import unittest

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


TEST_MODULES = [
    # 1. 3D U-Net Model & Weights
    ("3D U-Net Architecture", "federation.models.test_unet3d"),
    # 2. MRI Data Transforms & Partitioner
    ("Data Pipeline & Partitioner", "federation.datasets.test_mri_data"),
    # 3. Training Loops, Loss & Metrics
    ("Trainer & Loss Functions", "federation.models.test_trainer"),
    # 4. Flower Client (NumPyClient)
    ("FedMedClient Core", "federation.client.test_client"),
    # 5. Local Checkpointing & Diffing
    ("Client Checkpoint & Fallback", "federation.client.test_checkpoint"),
    # 6. Client CLI Runner & Hardware Inspection
    ("Client Runner & Hardware", "federation.client.test_run_client"),
    # 7. Baseline Server Coordinator
    ("FedAvg Server Baseline", "federation.server.test_server"),
    # 8. FedMedStrategy Weighted Aggregation
    ("FedMedStrategy Aggregation", "federation.server.test_strategy"),
    # 9. Global Model Checkpointing & Versioning
    ("Global Model Manager", "federation.server.test_model_manager"),
    # 10. Multi-Hospital Simulation Harness
    ("Simulation Testbed", "federation.test_simulate"),
    # 11. Parameter Encryption & DP
    ("TenSEAL & Differential Privacy", "federation.security.test_encryption"),
    # 12. Concurrency Locks & Synchronization
    ("Round Sync & Locks", "federation.server.test_sync_manager"),
    # 13. Auto-Dispatch & Auto-Aggregate Triggers
    ("Automated Triggers", "federation.server.test_triggers"),
    # 14. API Bridge & JSON Telemetry
    ("API Bridge & Telemetry", "federation.test_api_bridge"),
    # 15. Global Model Weight Encryption & HMAC Signatures
    ("Global Model Weight Encryption", "federation.security.test_global_encryption"),
    # 16. Complete FastAPI Backend & Telemetry Routes
    ("Production FastAPI Backend", "backend.test_backend"),
]


def run_master_test_suite() -> bool:
    print("\n" + "#" * 70)
    print("  FedMed - Master Verification & Full Test Suite")
    print(f"  Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print("#" * 70 + "\n")

    loader = unittest.TestLoader()
    suite_results = []
    total_tests_run = 0
    total_failures = 0
    total_errors = 0
    start_total_time = time.time()

    for idx, (display_name, module_name) in enumerate(TEST_MODULES, 1):
        print(f"[{idx:02d}/{len(TEST_MODULES):02d}] Running: {display_name} ({module_name})...", end="", flush=True)
        t0 = time.time()

        try:
            import io
            mod = __import__(module_name, fromlist=["*"])
            test_suite = loader.loadTestsFromModule(mod)
            err_stream = io.StringIO()
            runner = unittest.TextTestRunner(verbosity=1, stream=err_stream)
            result = runner.run(test_suite)

            elapsed = time.time() - t0
            tests_count = result.testsRun
            fails = len(result.failures)
            errs = len(result.errors)

            total_tests_run += tests_count
            total_failures += fails
            total_errors += errs

            passed = (fails == 0 and errs == 0)
            status_str = "PASSED" if passed else f"FAILED (Fails={fails}, Errs={errs})"

            suite_results.append({
                "name": display_name,
                "module": module_name,
                "tests": tests_count,
                "passed": passed,
                "fails": fails,
                "errs": errs,
                "time": elapsed,
                "error_details": err_stream.getvalue() if not passed else "",
            })

            print(f" -> {status_str} ({tests_count} tests in {elapsed:.2f}s)")
            if not passed:
                err_text = err_stream.getvalue().strip()
                if err_text:
                    for line in err_text.splitlines():
                        print(f"    {line}")

        except Exception as e:
            elapsed = time.time() - t0
            total_errors += 1
            suite_results.append({
                "name": display_name,
                "module": module_name,
                "tests": 0,
                "passed": False,
                "fails": 0,
                "errs": 1,
                "time": elapsed,
                "error_details": str(e),
            })
            print(f" -> ERROR: {e}")


    total_elapsed = time.time() - start_total_time

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Test Suite Name':<35} | {'Tests':<6} | {'Status':<10} | {'Time':<6}")
    print("-" * 70)
    for res in suite_results:
        status_lbl = "[OK] PASS" if res["passed"] else "[X] FAIL"
        print(f"{res['name']:<35} | {res['tests']:<6} | {status_lbl:<10} | {res['time']:<5.2f}s")
    print("=" * 70)

    print(f"\nFinal Summary: {total_tests_run} tests executed in {total_elapsed:.2f}s")
    print(f"  * Passed Suites : {sum(1 for r in suite_results if r['passed'])}/{len(TEST_MODULES)}")
    print(f"  * Failures      : {total_failures}")
    print(f"  * Errors        : {total_errors}")

    all_passed = (total_failures == 0 and total_errors == 0)
    if all_passed:
        print("\n>>> ALL 14 TEST SUITES PASSED PERFECTLY! <<<\n")
    else:
        print("\n" + "=" * 70)
        print("  FAILURES & ERRORS DETAIL")
        print("=" * 70)
        for res in suite_results:
            if not res["passed"]:
                print(f"\n--- [{res['name']}] ({res['module']}) ---")
                err_text = res.get("error_details", "").strip()
                if err_text:
                    print(err_text)
        print("=" * 70 + "\n")
        print(">>> SOME TESTS FAILED - CHECK DETAILS ABOVE <<<\n")

    return all_passed


if __name__ == "__main__":
    success = run_master_test_suite()
    sys.exit(0 if success else 1)
