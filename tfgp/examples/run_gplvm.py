import time

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

from tfgp.util.data import circle_data, gaussian_data, oilflow
from tfgp.util import PCA_reduce
from tfgp.model import GPLVM

if __name__ == "__main__":
    print("Generating data...")
    num_data = 100
    latent_dim = 2
    output_dim = 5
    y_obs, _, labels = oilflow(num_data, output_dim)
    x = PCA_reduce(y_obs, latent_dim)
    y = tf.convert_to_tensor(y_obs, dtype=tf.float32)

    print("Creating model...")
    m = GPLVM(y, latent_dim, x=x)

    print("Building graph...")
    loss = m.loss()
    learning_rate = 0.1
    with tf.name_scope("train"):
        trainable_vars = tf.trainable_variables()
        train = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=trainable_vars, name="RMSProp")
    with tf.name_scope("summary"):
        m.create_summaries()
        merged_summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    plt.axis([-5, 5, -5, 5])
    plt.ion()
    with tf.Session() as sess:
        log_dir = f"../../log/{time.strftime('%Y%m%d%H%M%S')}"
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 10000
        n_print = 200
        for i in range(n_iter):
            sess.run(train)
            if i % n_print == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                print(f"Step {i} - Loss: {sess.run(loss)}")
                x = sess.run(m.x)
                for c in np.unique(labels):
                    plt.scatter(*x[labels == c].T)
                plt.pause(0.05)
                plt.cla()
        x = sess.run(m.x)
        for c in np.unique(labels):
            plt.scatter(*x[labels == c].T)
        embed()
