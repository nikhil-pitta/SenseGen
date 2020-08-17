import data_utils
import model_utils
import model
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

data = data_utils.load_training_data()

# After every save, 3 files are generated
# replace the number at the end of both paths to total number of epochs - 1
import_path = './models/mdnmodel.ckpt-99.meta'
ckpt_path = './models/mdnmodel.ckpt-99'

seq_len = 2000
model_utils.reset_session_and_model()
true_data = data[0,:2000]
with tf.Session() as sess:
    test_config = model.ModelConfig()
    test_config.num_layers = 1
    test_config.batch_size = 1
    test_config.num_steps = 1
    test_model = model.MDNModel(test_config, True)
    test_model.is_training = False
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph(import_path)
    saver.restore(sess, ckpt_path)

    fake_data = test_model.predict(sess, seq_len)


# For showing real data vs fake data
time = [i+1 for i in range(len(fake_data))]

plt.plot(time, true_data)
plt.show()

plt.plot(time, fake_data)
plt.show()

fig, axes = plt.subplots(1,2, figsize=((14,8)))
axes[0].plot(true_data)
axes[0].set_title('True data')
axes[1].plot(fake_data)
axes[1].set_title('Fake data')


