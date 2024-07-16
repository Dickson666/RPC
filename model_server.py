import subprocess
import sys
import pkg_resources

with open("requirements.txt", 'r') as f:
    packages = f.readlines()
    for package in packages:
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(e, f"\nFailed to install {package}")

import torch
import grpc
from concurrent import futures
from PIL import Image
import io
import base64
import torchvision.transforms as transforms
import rpc.model_pb2 as model_pb2
import rpc.model_pb2_grpc as model_pb2_grpc
from model_class import model

trans = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor()
])

class ModelService(model_pb2_grpc.ModelServicer):
    def __init__(self) -> None:
        super().__init__()
        self.model = model()

    def mod(self, request, context):
        img = torch.tensor([])
        for i in request.indata:
            i_data = base64.b64decode(i)
            img_data = Image.open(io.BytesIO(i_data)).convert('RGB')
            img_tensor = trans(img_data)
            img = torch.cat([img, img_tensor.unsqueeze(0)], dim = 0)
        res = torch.tensor(self.model(img)).view(-1).tolist()
        return model_pb2.vector(outdata = res)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
    model_pb2_grpc.add_ModelServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()
    
if __name__ == '__main__':
    serve()