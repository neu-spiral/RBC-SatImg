import tensorflow as tf

class miou_binary(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, likelihood, sample_weight=None):
        likelihood = tf.where(likelihood>0.5, 1, 0)
        super().update_state(y_true, likelihood, sample_weight)

