import tensorflow as tf

print("Dispozitive:", tf.config.list_physical_devices())
with tf.device('/GPU:0'):
    a = tf.ones([1000, 1000])
    b = tf.ones([1000, 1000])
    c = tf.matmul(a, b)
print("Operațiune completă pe GPU")