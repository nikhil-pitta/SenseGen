import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def reset_session_and_model():
    """
    Resets the TensorFlow default graph and session.
    """
    tf.reset_default_graph()
    sess = tf.get_default_session()
    if sess:
        sess.close()

        
