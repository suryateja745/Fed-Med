from app.federated.client import HospitalClient
from app.federated.mock_hospitals import HOSPITALS


def test_three_hospitals_configured():
    assert set(HOSPITALS.keys()) == {
        "hospital-1",
        "hospital-2",
        "hospital-3",
    }


def test_hospital_ids_are_unique():
    ids = list(HOSPITALS.keys())
    assert len(ids) == len(set(ids))


def test_hospital_client_properties():
    client = HospitalClient("hospital-1")

    properties = client.get_properties({})

    assert properties["hospital_id"] == "hospital-1"
    assert properties["node_type"] == "hospital"
    assert properties["status"] == "connected"


def test_get_parameters():
    client = HospitalClient("hospital-1")

    parameters = client.get_parameters({})

    # Real UNet3D is represented as one flattened parameter vector.
    assert len(parameters) == 1
    assert parameters[0].size == 34886


def test_fit():
    client = HospitalClient("hospital-1")

    parameters = client.get_parameters({})

    updated, examples, metrics = client.fit(
        parameters,
        {},
    )

    # The encrypted 34,886-parameter vector is split into
    # multiple CKKS chunks for secure aggregation.
    assert len(updated) > 1
    assert examples == 1
    assert metrics["hospital_id"] == "hospital-1"
    assert metrics["dp_enabled"] is True
    assert metrics["encryption"] == "TenSEAL-CKKS"


def test_evaluate():
    client = HospitalClient("hospital-1")

    parameters = client.get_parameters({})

    loss, examples, metrics = client.evaluate(
        parameters,
        {},
    )

    assert loss >= 0.0
    assert loss < 10.0
    assert examples == 1
    assert metrics["hospital_id"] == "hospital-1"