import grpc

from app.proto import fedmed_pb2
from app.proto import fedmed_pb2_grpc


GRPC_SERVER = "localhost:50051"


def create_channel():
    """Create a gRPC channel to the FedMed backend."""
    return grpc.insecure_channel(GRPC_SERVER)


def health_check(hospital_id="frontend-hospital"):
    """
    Check whether the FedMed backend is healthy.

    Returns a dictionary so Streamlit can easily use the result.
    """

    channel = create_channel()

    try:
        stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

        response = stub.HealthCheck(
            fedmed_pb2.HealthRequest(
                hospital_id=hospital_id
            ),
            timeout=5,
        )

        return {
            "success": response.status.lower() == "healthy",
            "status": response.status,
            "message": response.message,
        }

    except grpc.RpcError as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Backend connection failed: {e.details()}",
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
        }

    finally:
        channel.close()


def register_hospital(hospital_id, hospital_name, location):
    """
    Register a hospital with the FedMed backend.

    Returns a dictionary containing success and message.
    """

    channel = create_channel()

    try:
        stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

        response = stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id=hospital_id,
                hospital_name=hospital_name,
                location=location,
            ),
            timeout=5,
        )

        return {
            "success": response.success,
            "status": "registered" if response.success else "error",
            "message": response.message,
        }

    except grpc.RpcError as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Backend connection failed: {e.details()}",
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
        }

    finally:
        channel.close()