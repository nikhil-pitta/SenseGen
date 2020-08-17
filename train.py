import data_utils
import model_utils
import model
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

data = data_utils.load_training_data()

# The number of epochs the algorithm will run
num_epochs = 100
# After this many epochs, the algorithm will save itself under /models
# Ex: if save_after = 100, the algorithm will save itself every 100 epochs
save_after = 1

model_utils.reset_session_and_model()
with tf.Session() as sess:
    train_config = model.ModelConfig()
    test_config = model.ModelConfig()
    train_config.learning_rate = 0.0003
    train_config.num_layers = 1 
    test_config.num_layers = 1
    test_config.batch_size = 1
    test_config.num_steps = 1
    loader = data_utils.DataLoader(data=data,batch_size=train_config.batch_size, num_steps=train_config.num_steps)
    train_model = model.MDNModel(train_config, True)
    test_model = model.MDNModel(test_config, False)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for idx in range(num_epochs):
        epoch_loss = train_model.train_for_epoch(sess, loader)
        print(idx, ' ', epoch_loss)
        if (idx+1) % save_after == 0:
            saver.save(sess, './models/mdnmodel.ckpt', global_step=idx)