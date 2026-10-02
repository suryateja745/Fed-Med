from concurrent import futures

import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


class FedMedService(fedmed_pb2_grpc.FedMedServiceServicer):
    def __init__(self):
        self.hospitals = {}

    def HealthCheck(self, request, context):
        try:
            hospital_id = request.hospital_id.strip()

            if not hospital_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Hospital ID is required",
                )

            return fedmed_pb2.HealthResponse(
                status="healthy",
                message=f"Hospital {hospital_id} connected successfully",
            )

        except grpc.RpcError:
            raise

        except Exception:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Internal server error during health check",
            )

    def RegisterHospital(self, request, context):
        hospital_id = request.hospital_id.strip()
        hospital_name = request.hospital_name.strip()
        location = request.location.strip()

        if not hospital_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Hospital ID is required",
            )

        if not hospital_name:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Hospital name is required",
            )

        if not location:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Hospital location is required",
            )

        if hospital_id in self.hospitals:
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"Hospital {hospital_id} is already registered",
            )

        try:
            self.hospitals[hospital_id] = {
                "hospital_name": hospital_name,
                "location": location,
                "status": "online",
            }

            return fedmed_pb2.RegisterHospitalResponse(
                success=True,
                message=f"Hospital {hospital_id} registered successfully",
            )

        except Exception:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Internal server error while registering hospital",
            )

    def GetHospitalStatus(self, request, context):
        hospital_id = request.hospital_id.strip()

        if not hospital_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Hospital ID is required",
            )

        hospital = self.hospitals.get(hospital_id)

        if hospital is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Hospital {hospital_id} is not registered",
            )


        try:
            return fedmed_pb2.GetHospitalStatusResponse(
                success=True,
                hospital_id=hospital_id,
                hospital_name=hospital["hospital_name"],
                location=hospital["location"],
                status=hospital["status"],
                message=f"Hospital {hospital_id} is online",
            )
        except Exception:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Internal server error while getting hospital status",
            )

    def GetAllHospitals(self, request, context):
        try:
            hospitals = []

            for hospital_id, hospital in self.hospitals.items():
                hospitals.append(
                    fedmed_pb2.Hospital(
                        hospital_id=hospital_id,
                        hospital_name=hospital["hospital_name"],
                        location=hospital["location"],
                        status=hospital["status"],
                    )
                )

            return fedmed_pb2.GetAllHospitalsResponse(
                success=True,
                hospitals=hospitals,
                message=f"{len(hospitals)} hospital(s) retrieved successfully",
            )
        except Exception:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Internal server error while getting hospitals",
            )

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    fedmed_pb2_grpc.add_FedMedServiceServicer_to_server(
        FedMedService(),
        server,
    )

    server.add_insecure_port("[::]:50051")

    server.start()

    print("FedMed gRPC server started on port 50051")

    try:
        server.wait_for_termination()

    except KeyboardInterrupt:
        print("\nFedMed gRPC server stopped.")

        server.stop(0)


if __name__ == "__main__":
    serve()
