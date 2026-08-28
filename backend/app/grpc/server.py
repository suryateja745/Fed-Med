from concurrent import futures

import grpc

from app.grpc import fedmed_pb2
from app.grpc import fedmed_pb2_grpc


class FedMedService(fedmed_pb2_grpc.FedMedServiceServicer):

    def HealthCheck(self, request, context):
        return fedmed_pb2.HealthResponse(
            status="healthy",
            message=f"Hospital {request.hospital_id} connected successfully",
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

    server.wait_for_termination()


if __name__ == "__main__":
    serve()