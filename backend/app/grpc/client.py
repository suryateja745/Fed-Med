import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


def check_server(hospital_id: str):
    channel = grpc.insecure_channel("localhost:50051")

    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    request = fedmed_pb2.HealthRequest(
        hospital_id=hospital_id
    )

    response = stub.HealthCheck(request)

    print("Status:", response.status)
    print("Message:", response.message)


if __name__ == "__main__":
    check_server("hospital-1")