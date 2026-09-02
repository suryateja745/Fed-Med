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
            hospital_id="test-health-hospital"
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
            hospital_id="test-register-hospital",
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

    register_response = stub.RegisterHospital(
        fedmed_pb2.RegisterHospitalRequest(
            hospital_id="status-test-hospital",
            hospital_name="Status Test Hospital",
            location="Hyderabad",
        )
    )

    assert register_response.success is True

    response = stub.GetHospitalStatus(
        fedmed_pb2.GetHospitalStatusRequest(
            hospital_id="status-test-hospital"
        )
    )

    assert response.success is True
    assert response.hospital_id == "status-test-hospital"
    assert response.hospital_name == "Status Test Hospital"
    assert response.location == "Hyderabad"
    assert response.status == "online"
    assert "is online" in response.message

    channel.close()


def test_register_empty_hospital_id():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id="",
                hospital_name="Test Hospital",
                location="Hyderabad",
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
                hospital_id="empty-name-hospital",
                hospital_name="",
                location="Hyderabad",
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
                hospital_id="empty-location-hospital",
                hospital_name="Test Hospital",
                location="",
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    channel.close()


def test_duplicate_hospital_registration():
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    request = fedmed_pb2.RegisterHospitalRequest(
        hospital_id="duplicate-hospital",
        hospital_name="Duplicate Hospital",
        location="Hyderabad",
    )

    first_response = stub.RegisterHospital(request)

    assert first_response.success is True

    with pytest.raises(grpc.RpcError) as error:
        stub.RegisterHospital(request)

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