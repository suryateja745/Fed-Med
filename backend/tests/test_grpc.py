import grpc

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
            location="Mumbai"
        )
    )

    assert response.success is True
    assert "registered successfully" in response.message

    channel.close()