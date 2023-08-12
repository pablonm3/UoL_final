import redis
import tensorflow as tf
from configReader import get_config

def serialize_tensor(tensor_3d):

    # List to store serialized 2D tensors
    serialized_tensors = []

    # Iterate through the first dimension of the 3D tensor
    for i in range(tensor_3d.shape[0]):
        # Extract the 2D tensor at the current index
        tensor_2d = tensor_3d[i]

        # Serialize the 2D tensor using TensorFlow's method
        serialized_tensor = tf.io.serialize_tensor(tensor_2d).numpy()

        # Append the serialized tensor to the list
        serialized_tensors.append(serialized_tensor)
    return serialized_tensors

def deserialize_tensor(serialized_tensors):
    # List to store deserialized 2D tensors
    deserialized_tensors = []

    # Iterate through the serialized tensors
    for serialized_tensor in serialized_tensors:
        # Deserialize the tensor using TensorFlow's method
        deserialized_tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

        # Append the deserialized tensor to the list
        deserialized_tensors.append(deserialized_tensor)

    # Stack the 2D tensors along the first dimension to form a 3D tensor
    tensor_3d_reconstructed = tf.stack(deserialized_tensors)
    return tensor_3d_reconstructed

class RedisCache:
    def __init__(self, clear_cache, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)
        if clear_cache:
            self.clear()

    def set(self, key, value):
        strings_list = serialize_tensor(value)
        for value in strings_list:
            self.redis.rpush(key, value) # rpush appends latest elements at the end of list. maintaining order of insertion

    def get(self, key):
        result = self.redis.lrange(key, 0, -1)
        #strings_list = [item.decode('utf-8') for item in result]

        tensor_3d = deserialize_tensor(result)
        return tensor_3d

    def delete(self, key):
        self.redis.delete(key)

    def exists(self, key):
        return self.redis.exists(key)

    def clear(self):
        print("[WARNING] clearing redis cache...")
        # clear all keys from db
        self.redis.flushdb()
