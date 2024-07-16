import grpc
import rpc.model_pb2 as model_pb2
import rpc.model_pb2_grpc as model_pb2_grpc
import torch
import base64 
import os
import io
from PIL import Image

def rand_base64(n = 5, isize = 114):
    img = []
    for _ in range(n):
        randbyte = os.urandom(isize)
        b64_img = base64.b64encode(randbyte).decode('utf-8')
        img.append(b64_img)
    return img

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_pb2_grpc.ModelStub(channel)

    ip = '1.jpg'
    with open(ip, 'rb') as imf:
        id = imf.read()
    ib = io.BytesIO(id)
    bi = base64.b64encode(ib.read()).decode('utf-8')
    data = [bi]
    
    # data = rand_base64()
    ii = base64.b64decode(data[0])
    img = Image.open(io.BytesIO(ii)).convert('RGB')
    indata = model_pb2.img(indata = data, img_num = 5)

    res = stub.mod(indata)
    print(type(res))
    res = torch.tensor(res.outdata).view(5, -1).tolist()

    print(res)

if __name__ == '__main__':
    run()