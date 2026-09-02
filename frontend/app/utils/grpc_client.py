import grpc

from app.proto import fedmed_pb2
from app.proto import fedmed_pb2_grpc

GRPC_SERVER = "localhost:50051"


def health_check(hospital_id: str = "frontend-client"):
    channel = grpc.insecure_channel(GRPC_SERVER)
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        request = fedmed_pb2.HealthRequest(
            hospital_id=hospital_id
        )

        response = stub.HealthCheck(
            request,
            timeout=5
        )

        return {
            "success": True,
            "status": response.status,
            "message": response.message,
        }

    except grpc.RpcError as error:
        return {
            "success": False,
            "status": error.code().name,
            "message": error.details(),
        }

    finally:
        channel.close()


def register_hospital(
    hospital_id: str,
    hospital_name: str,
    location: str,
):
    channel = grpc.insecure_channel(GRPC_SERVER)
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        request = fedmed_pb2.RegisterHospitalRequest(
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            location=location,
        )

        response = stub.RegisterHospital(
            request,
            timeout=5
        )

        return {
            "success": response.success,
            "message": response.message,
        }

    except grpc.RpcError as error:
        return {
            "success": False,
            "status": error.code().name,
            "message": error.details(),
        }

    finally:
        channel.close()


def get_hospital_status(hospital_id: str):
    channel = grpc.insecure_channel(GRPC_SERVER)
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        request = fedmed_pb2.GetHospitalStatusRequest(
            hospital_id=hospital_id
        )

        response = stub.GetHospitalStatus(
            request,
            timeout=5
        )

        return {
            "success": response.success,
            "hospital_id": response.hospital_id,
            "hospital_name": response.hospital_name,
            "location": response.location,
            "status": response.status,
            "message": response.message,
        }

    except grpc.RpcError as error:
        return {
            "success": False,
            "status": error.code().name,
            "message": error.details(),
        }

    finally:
        channel.close()