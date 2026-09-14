from concurrent import futures

import grpc
import pytest

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc
from app.grpc.server import FedMedService


@pytest.fixture
def grpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    fedmed_pb2_grpc.add_FedMedServiceServicer_to_server(
        FedMedService(),
        server,
    )

    port = server.add_insecure_port("localhost:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel

    channel.close()
    server.stop(0).wait()


def test_health_check(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    response = stub.HealthCheck(
        fedmed_pb2.HealthRequest(
            hospital_id="test-hospital-1"
        )
    )

    assert response.status == "healthy"
    assert "connected successfully" in response.message


def test_register_hospital(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    response = stub.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="test-hospital-1",
            hospital_name="Test Hospital",
            location="Mumbai",
        )
    )

    assert response.success is True
    assert "registered successfully" in response.message


def test_get_hospital_status(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    stub.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="test-hospital-1",
            hospital_name="Test Hospital",
            location="Mumbai",
        )
    )

    response = stub.GetHospitalStatus(
        fedmed_pb2.GetHospitalStatusRequest(
            hospital_id="test-hospital-1"
        )
    )

    assert response.success is True
    assert response.hospital_id == "test-hospital-1"
    assert response.hospital_name == "Test Hospital"
    assert response.location == "Mumbai"
    assert response.status == "online"


def test_register_empty_hospital_id(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="",
                hospital_name="Test Hospital",
                location="Mumbai",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_register_empty_hospital_name(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="test-hospital-2",
                hospital_name="",
                location="Mumbai",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_register_empty_location(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="test-hospital-3",
                hospital_name="Test Hospital",
                location="",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_duplicate_hospital_registration(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    hospital_id = "duplicate-hospital"

    stub.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id=hospital_id,
            hospital_name="Duplicate Hospital",
            location="Delhi",
        )
    )

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id=hospital_id,
                hospital_name="Duplicate Hospital",
                location="Delhi",
            )
        )

    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_unknown_hospital_status(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    with pytest.raises(grpc.RpcError) as error:
        stub.GetHospitalStatus(
            fedmed_pb2.GetHospitalStatusRequest(
                hospital_id="unknown-hospital"
            )
        )

    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_empty_hospital_status_id(grpc_server):
    stub = fedmed_pb2_grpc.FedMedServiceStub(grpc_server)

    with pytest.raises(grpc.RpcError) as error:
        stub.GetHospitalStatus(
            fedmed_pb2.GetHospitalStatusRequest(
                hospital_id=""
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_get_all_hospitals():
    service = FedMedService()

    class FakeContext:
        def abort(self, code, details):
            raise RuntimeError(f"{code}: {details}")

    context = FakeContext()

    service.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="hospital-001",
            hospital_name="Hospital One",
            location="Hyderabad",
        ),
        context,
    )

    service.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="hospital-002",
            hospital_name="Hospital Two",
            location="Bengaluru",
        ),
        context,
    )

    response = service.GetAllHospitals(
        fedmed_pb2.GetAllHospitalsRequest(),
        context,
    )

    assert response.success is True
    assert len(response.hospitals) == 2

    hospital_ids = {
        hospital.hospital_id
        for hospital in response.hospitals
    }

    assert hospital_ids == {
        "hospital-001",
        "hospital-002",
    }


def test_get_all_hospitals_empty():
    service = FedMedService()

    class FakeContext:
        def abort(self, code, details):
            raise RuntimeError(f"{code}: {details}")

    context = FakeContext()

    response = service.GetAllHospitals(
        fedmed_pb2.GetAllHospitalsRequest(),
        context,
    )

    assert response.success is True
    assert len(response.hospitals) == 0
    assert "0 hospital(s)" in response.message