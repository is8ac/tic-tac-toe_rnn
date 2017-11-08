import tensorflow as tf

board = tf.placeholder(tf.float32, shape=[2, 3, 3], name="input")
empty_board = tf.zeros_like(board, dtype=tf.float32)
weights = tf.Variable(tf.random_normal([9, 18], stddev=0.5), name="weights")
mul = tf.matmul(weights, tf.reshape(empty_board, [18, 1]))
print("mul:", mul.shape)
biases = tf.Variable(tf.zeros([9, 1]), name="biases")
print("biases:", biases.shape)
cell_output = tf.add(mul, biases)
move = tf.reshape(cell_output, [3, 3])

tf.train.write_graph(tf.get_default_graph().as_graph_def(), '', 'model.pb.txt', as_text=True)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(move))
