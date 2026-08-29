import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


def check_server(hospital_id):
    channel = grpc.insecure_channel("localhost:50051")

    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    request = fedmed_pb2.HealthRequest(
        hospital_id=hospital_id
    )

    response = stub.HealthCheck(request)

    print("Status:", response.status)
    print("Message:", response.message)


def register_hospital(hospital_id, hospital_name):
    channel = grpc.insecure_channel("localhost:50051")

    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    request = fedmed_pb2.RegisterHospitalRequest(
        hospital_id=hospital_id,
        hospital_name=hospital_name,
    )

    response = stub.RegisterHospital(request)

    print("Registration Status:", response.status)
    print("Registration Message:", response.message)


if __name__ == "__main__":
    check_server("hospital-1")
    register_hospital("hospital-1", "Hospital One")