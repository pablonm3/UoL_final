import redis
import tensorflow as tf

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)

    def set(self, key, value):
        serialized_value = tf.io.serialize_tensor(value).numpy()
        self.redis.set(key, serialized_value)

    def get(self, key):
        value = self.redis.get(key)
        if value is not None:
            value = tf.io.parse_tensor(value, out_type=tf.float32)
        return value

    def delete(self, key):
        self.redis.delete(key)

    def exists(self, key):
        return self.redis.exists(key)

    def clear(self):
        print("[WARNING] clearing redis cache...")
        # clear all keys from db
        self.redis.flushdb()
