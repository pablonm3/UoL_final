import os

import tensorflow as tf
import shutil

def delete_contents(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
class TensorCache:
    def __init__(self, clear_cache):
        if clear_cache:
            self.clear()

    def set(self, key, value):
        filename = "cache_files/" + key + ".tf"
        serialized_tensor = tf.io.serialize_tensor(value)
        tf.io.write_file(filename, serialized_tensor)
    def get(self, key):
        filename = "cache_files/" + key + ".tf"
        serialized_tensor = tf.io.read_file(filename)
        return tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

    def exists(self, key):
        filename = "cache_files/" + key + ".tf"
        return os.path.exists(filename)

    def clear(self):
        print("[WARNING] clearing tensor cache...")
        # clear all keys from db
        delete_contents("cache_files")
