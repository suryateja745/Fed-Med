from concurrent import futures

import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


class FedMedService(fedmed_pb2_grpc.FedMedServiceServicer):
    def __init__(self):
        self.hospitals = {}

    def HealthCheck(self, request, context):
        return fedmed_pb2.HealthResponse(
            status="healthy",
            message=f"Hospital {request.hospital_id} connected successfully",
        )

    def RegisterHospital(self, request, context):
        if not request.hospital_id.strip():
            return fedmed_pb2.RegisterHospitalResponse(
                success=False,
                message="Hospital ID is required",
            )

        if not request.hospital_name.strip():
            return fedmed_pb2.RegisterHospitalResponse(
                success=False,
                message="Hospital name is required",
            )

        if not request.location.strip():
            return fedmed_pb2.RegisterHospitalResponse(
                success=False,
                message="Hospital location is required",
            )

        if request.hospital_id in self.hospitals:
            return fedmed_pb2.RegisterHospitalResponse(
                success=False,
                message=f"Hospital {request.hospital_id} is already registered",
            )

        self.hospitals[request.hospital_id] = {
            "hospital_name": request.hospital_name,
            "location": request.location,
        }

        return fedmed_pb2.RegisterHospitalResponse(
            success=True,
            message=f"Hospital {request.hospital_id} registered successfully",
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    fedmed_pb2_grpc.add_FedMedServiceServicer_to_server(
        FedMedService(),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("FedMed gRPC server started on port 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()