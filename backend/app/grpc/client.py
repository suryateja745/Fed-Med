import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


SERVER_ADDRESS = "localhost:50051"


def create_channel():
    return grpc.insecure_channel(SERVER_ADDRESS)


def check_server(hospital_id: str):
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        response = stub.HealthCheck(
            fedmed_pb2.HealthRequest(
                hospital_id=hospital_id
            )
        )

        print("Status:", response.status)
        print("Message:", response.message)

        return response

    except grpc.RpcError as error:
        print("HealthCheck failed")
        print("Code:", error.code())
        print("Details:", error.details())

    finally:
        channel.close()


def register_hospital(
    hospital_id: str,
    hospital_name: str,
    location: str,
):
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        response = stub.RegisterHospital(
            fedmed_pb2.RegisterHospitalRequest(
                hospital_id=hospital_id,
                hospital_name=hospital_name,
                location=location,
            )
        )

        print("Registration success:", response.success)
        print("Registration message:", response.message)

        return response

    except grpc.RpcError as error:
        print("Hospital registration failed")
        print("Code:", error.code())
        print("Details:", error.details())

    finally:
        channel.close()


def get_hospital_status(hospital_id: str):
    channel = create_channel()
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        response = stub.GetHospitalStatus(
            fedmed_pb2.GetHospitalStatusRequest(
                hospital_id=hospital_id
            )
        )

        print("Status check success:", response.success)
        print("Hospital ID:", response.hospital_id)
        print("Hospital Name:", response.hospital_name)
        print("Location:", response.location)
        print("Hospital Status:", response.status)
        print("Message:", response.message)

        return response

    except grpc.RpcError as error:
        print("Hospital status check failed")
        print("Code:", error.code())
        print("Details:", error.details())

    finally:
        channel.close()


if __name__ == "__main__":
    check_server("hospital-1")

    register_hospital(
        "hospital-1",
        "Hospital One",
        "Hyderabad",
    )
    get_hospital_status("hospital-1")


def get_all_hospitals():
    channel = grpc.insecure_channel("localhost:50051")
    stub = fedmed_pb2_grpc.FedMedServiceStub(channel)

    try:
        request = fedmed_pb2.GetAllHospitalsRequest()

        response = stub.GetAllHospitals(
            request,
            timeout=5,
        )

        print("Get all hospitals success:", response.success)
        print("Message:", response.message)
        print("Total hospitals:", len(response.hospitals))

        for hospital in response.hospitals:
            print(
                "Hospital:",
                hospital.hospital_id,
                "| Name:",
                hospital.hospital_name,
                "| Location:",
                hospital.location,
                "| Status:",
                hospital.status,
            )

        return response

    except grpc.RpcError as error:
        print("Get all hospitals failed")
        print("Status code:", error.code())
        print("Details:", error.details())

    finally:
        channel.close()
