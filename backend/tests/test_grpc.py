import grpc
import pytest

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


def create_channel():
    return grpc.insecure_channel("localhost:50051")


def test_health_check():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    response = stub.HealthCheck(
        fedmed_pb2.HealthRequest(
            hospital_id="test-hospital-1"
        )
    )

    assert response.status == "healthy"
    assert "connected successfully" in response.message

    channel.close()


def test_register_hospital():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    response = stub.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="test-hospital-1",
            hospital_name="Test Hospital",
            location="Mumbai",
        )
    )

    assert response.success is True
    assert "registered successfully" in response.message

    channel.close()


def test_get_hospital_status():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

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

    channel.close()


def test_register_empty_hospital_id():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="",
                hospital_name="Test Hospital",
                location="Mumbai",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    channel.close()


def test_register_empty_hospital_name():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="test-hospital-2",
                hospital_name="",
                location="Mumbai",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    channel.close()


def test_register_empty_location():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="test-hospital-3",
                hospital_name="Test Hospital",
                location="",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    channel.close()


def test_duplicate_hospital_registration():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

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

    channel.close()


def test_unknown_hospital_status():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.GetHospitalStatus(
            fedmed_pb2.GetHospitalStatusRequest(
                hospital_id="unknown-hospital"
            )
        )

    assert error.value.code() == grpc.StatusCode.NOT_FOUND

    channel.close()


def test_empty_hospital_status_id():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.GetHospitalStatus(
            fedmed_pb2.GetHospitalStatusRequest(
                hospital_id=""
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    channel.close()