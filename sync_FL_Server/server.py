import os
import grpc
import base64
import sys
import warnings
from time import sleep, time
import functions_pb2
import functions_pb2_grpc
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json

# Initialize logging FIRST
from utility.logging_config import initialize_logger, get_logger
logger = initialize_logger("FIDEL-Server", "logs/server_verbose.log")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, Reshape, MaxPooling1D, Flatten
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical
logger.info("All imports completed successfully")

# gRPC Channel Configuration
logger.info("Configuring gRPC channels to client devices...")
try:
    channel01 = grpc.insecure_channel('192.168.0.204:8080')
    channel02 = grpc.insecure_channel('192.168.0.192:8080')
    channel03 = grpc.insecure_channel('192.168.0.106:8080')
    channel04 = grpc.insecure_channel('192.168.0.251:8080')
    channel05 = grpc.insecure_channel('192.168.0.23:8080')
    logger.info("All gRPC channels created successfully")
except Exception as e:
    logger.error(f'Failed to create gRPC channels: {str(e)}', exc_info = True)
    sys.exit(1)

# gRPC Stub Configuration
logger.info("Creating gRPC stubs for client communication...")
try:
    stub01 = functions_pb2_grpc.FederatedAppStub(channel01)
    stub02 = functions_pb2_grpc.FederatedAppStub(channel02)
    stub03 = functions_pb2_grpc.FederatedAppStub(channel03)
    stub04 = functions_pb2_grpc.FederatedAppStub(channel04)
    stub05 = functions_pb2_grpc.FederatedAppStub(channel05)
except Exception as e:
    logger.error(f'Failed to create gRPC stubs: {str(e)}', exc_info = True)
    sys.exit(1)

stubs = [stub01, stub02, stub03, stub04, stub05]
n = len(stubs)
iteration = 0

logger.info(f'Server configured for {n} client devices')
logger.info("=" * 80)

# Function to convert model(h5) to string
def encode_file(file_name):
    '''
    Encode the model file to base64 for transmission
    '''
    logger.func_enter("encode_file", file_name = file_name)
    try:
        file_path = 'Models/' + file_name
        logger.debug(f'Reading model file: {file_path}')
        with open(file_path, 'rb') as file:
            file_size = os.path.getsize(file_path)
            logger.debug(f'File size: {file_size} bytes')
            encoded_string = base64.b64encode(file.read())
        logger.debug(f'Encoded size: {len(encoded_string)} bytes')
        logger.func_exit("encode_file", result = encoded_string[:50])
        return encoded_string
    except Exception as e:
        logger.error(f'Failed to encode file {file_name}: {str(e)}', exc_info = True)
        raise












