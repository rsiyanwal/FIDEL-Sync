import pandas as pd
import numpy as np
import json 
import base64
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import time

# Initialize logging FIRST
from utility.logging_config import initialize_logger, get_logger
logger = initialize_logger('FIDEL-Function', 'logs/server_verbose.log')
logger.info('Client functions module loaded with verbose logging')

def saveLearntMetrics(file_name, score):
    '''
    Saving training metrics to a file
    '''
    logger.func_enter('saveLearntMetrics', file_name = file_name, loss = f'{score[0]:.6f}', accuracy = f'{score[1]:.4f}')
    try:
        logger.debug(f'Metrics: Loss: {score[0]:.6f}, Accuracy: {score[1]:.4f}')
        with open(file_name, 'r+') as f:
            trainMetrics = json.load(f)
            # old_acc_count = len(trainMetrics['accuracy'])
            trainMetrics['accuracy'].append(float(score[1]))
            trainMetrics['loss'].append(float(score[0]))
            f.seek(0)
            f.truncate()
            f.write(json.dumps(trainMetrics))
        logger.info(f"Metrics saved to {file_name} - Total records: {len(trainMetrics['accuracy'])}")
        logger.func_exit('saveLearntMetrics', result = 'success')
    except Exception as e:
        logger.error(f'Faled to save metrics to {file_name}: {str(e)}', exc_info = True)
        raise

def FetchClientResults():
    '''
    Fetch results from the local client
    '''
    logger.func_enter('FetchClientResults')
    logger.info('Fetching client results...')
    result = 'result from client'
    logger.func_exit('FetchClientResults', result = result)
    return result

'''
1. Server sends global model.
2. Client decodes it.
3. Client evaluates the global model.
4. Client trains on local private data.
5. Client evaluates updated model.
6. Client sends updated model back.
'''
def Train(modelString):
    '''
    Train a local model based on private data
    '''
    logger.info('=' * 80)
    logger.info('Starting local model training')
    logger.info('=' * 80)

    flag = False # Stream data or normal FL
    start_time = time.time()

    try:
        decode_model = base64.b64decode(modelString)
        logger.func_enter('Train', model_size = len(decode_model))
        # Step 1: Decode and Load model
        logger.step(1, 'Decoding received model')
        try:
            with open('data/current_model.h5', 'wb') as file:
                file.write(decode_model)
                logger.debug(f'Model decoded and saved - Size: {len(decode_model)} bytes')
        except Exception as e:
            logger.error(f'Failed to decode/save model: {str(e)}', exc_info = True)
            raise

        logger.step(2, 'Loading model architecture')
        try:
            model = load_model('data/current_model.h5')
            logger.info('Model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}', exc_info = True)
            raise

        # Step 3: Prepare training data
        X, y = [], []
        if flag: 
            logger.step(3, 'Continuous learning mode (streaming data)')
            continuousTrainingBatchSize = 60

            # Reading index to simulate a continuous learning
            logger.debug('Reading data index from file')
            currentIndex = 0
            try:
                with open('data/indexFile.txt', 'r+') as f:
                    fileIndex = json.load(f)
                    currentIndex = fileIndex['index']
                    logger.debug(f'Current index: {currentIndex}')
            except Exception as e:
                logger.error(f'Failed to read index file: {str(e)}', exc_info = True)
                raise

            try:
                data = pd.read_csv('data/data.csv')
                totalRowCount = data.shape[0]
                logger.debug(f'Dataset shape: {data.shape} - Total rows: {totalRowCount}')
                nextIndex = currentIndex + continuousTrainingBatchSize if currentIndex + continuousTrainingBatchSize < totalRowCount else totalRowCount
                logger.debug(f'Preparing data from index {currentIndex} to {nextIndex}')
                X = data.iloc[currentIndex:nextIndex, 1:-1].values
                y = data.iloc[currentIndex:nextIndex, -1].values
                y = to_categorical(y)
                logger.info(f'Data prepared for continuous learning - X shape: {X.shape}, y shape: {y.shape}')

                # Update index
                if nextIndex == totalRowCount:
                    nextIndex = 0
                with open('data/indexFile.txt', 'w') as f:
                    index = {'index': nextIndex}
                    f.write(json.dumps(index))
                logger.debug(f'Index updated to {nextIndex}')
            except Exception as e:
                logger.error(f'Failed to prepare continuous data: {str(e)}', exc_info = True)
                raise
        else:
            logger.step(3, 'Standard federated learning mode (batch data)')
            try:
                data = pd.read_csv('data/data.csv')
                logger.debug(f'Dataset loaded from CSV - shape: {data.shape}')
                X = data.iloc[:, 1:-1].values
                y = data.iloc[:, -1].values
                y = to_categorical(y)
                logger.info(f'Data loaded - X shape: {X.shape}, y shape: {y.shape}')
            except Exception as e:
                logger.error(f'Failed to load data: {str(e)}', exc_info = True)
                raise

        # Step 4: Evaluate the Global model
        logger.step(4, 'Evaluating received global model')
        try:
            score = model.evaluate(X, y, verbose = 0)
            logger.info(f'Global model evaluation - Loss: {score[0]:.6f}, Accuracy: {score[1]:.4f}')
            saveLearntMetrics('data/metrics.txt', score)
        except Exception as e:
            logger.error(f'Failed to evaluate global model: {str(e)}', exc_info = True)
            raise

        # Step 5: Train local model
        logger.step(5, 'Starting model training')
        try:
            logger.info('Training configuration: batch_size = 32, epochs = 16, shuffle = True')
            logger.wait('Waiting for local model to train')
            model.fit(X, y, batch_size = 32, epochs = 16, shuffle = True, verbose = 0)
            logger.info('Model training complete')
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f'Training failed after {elapsed:.2f}s: {str(e)}', exc_info = True)
            logger.warning('Returning original model due to training failure')
            return modelString

        # Step 6: Evaluate the trained model
        logger.step(6, 'Evaluating the trained local model')
        try:
            score = model.evaluate(X, y, verbose = 0)
            logger.info(f'Local model evaluation - Loss: {score[0]:.6f}, Accuracy: {score[1]:.4f}')
            saveLearntMetrics('data/localMetrics.txt', score)
        except Exception as e:
            logger.error(f'Failed to evaluate trained model: {str(e)}', exc_info = True)
            raise

        # Step 7: Save trained model
        logger.step(7, 'Saving trained model')
        try:
            model.save('data/model.h5')
            logger.debug('Model saved to data/model.h5')
        except Exception as e:
            logger.error(f'Failed to save model: {str(e)}', exc_info = True)
            raise

        # Step 8: Encode model for transmission
        logger.step(8, 'Encoding model for transmission back to the server')
        try:
            with open('data/model.h5', 'rb') as file:
                file_content = file.read()
                encoded_string = base64.b64encode(file_content)
                logger.debug(f'Model encoded - Original: {len(file_content)} bytes, Encoded: {len(encoded_string)} bytes')
        except Exception as e:
            logger.error(f'Failed to encode model: {str(e)}', exc_info = True)
            raise

        elapsed = time.time() - start_time
        logger.info(f'Local training completed successfully in {elapsed:2f}s')
        logger.checkpoint('Training', f'Completed in {elapsed:.2f}s, Model size: {len(encoded_string)} bytes')
        logger.info('=' * 80)
        logger.func_exit('Train', result = 'encoded_model')

        #return encoded_string
        return encoded_string.decode("utf-8")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.critical(f'Training function FAILED after {elapsed:.2f}s: {str(e)}', exc_info = True)
        logger.func_exit('Train', result = 'error_returned_original_model')
        raise

def InitializeClient():
    '''
    Initialize client data structures
    '''
    logger.func_enter('InitializeClient')
    logger.info('Initializing client data structures...')

    try:
        metric = {'accuracy': [], 'loss': []}
        index = {'index': 0}
        logger.step(1, 'Creating index file')
        try:
            with open('data/indexFile.txt', 'w') as f:
                f.write(json.dumps(index))
                logger.debug('Index file created: data/indexFile.txt')
        except Exception as e:
            logger.error(f'Failed to create index file: {str(e)}', exc_info = True)
            return 1

        logger.step(2, 'Creating global metrics file')
        try:
            with open('data/metrics.txt', 'w') as f:
                f.write(json.dumps(metric))
                logger.debug('Global metrics file created: data/metrics.txt')
        except Exception as e:
            logger.error(f'Failed to create metrics file: {str(e)}', exc_info = True)
            return 1

        logger.step(3, 'Creating local metrics file')
        try:
            with open('data/localMetrics.txt', 'w') as f:
                f.write(json.dumps(metric))
                logger.debug('Local metric file created: data/localMetrics.txt')
        except Exception as e:
            logger.error(f'Failed to create local metrics file: {str(e)}', exc_info = True)
            return 1

        logger.info('Client initialization completed successfully.')
        logger.checkpoint('Client Init', 'All data structures created')
        logger.func_exit('InitializeClient', result = '0')
        return 0
    except Exception as e:
        logger.error(f'Client initialization failed: {str(e)}', exc_info = True)
        logger.func_exit('InitializeClient', result = '-1')
        return 1
