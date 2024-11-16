from __future__ import absolute_import, division, print_function
import tensorflow.compat.v1 as tf
import numpy as np
import os

tf.disable_eager_execution()
tf.reset_default_graph()

def normalize(data):
    min_values, max_values = np.min(data, axis=0), np.max(data, axis=0)
    scaled_data = (data - min_values) / (max_values - min_values + 1e-7)
    return scaled_data, min_values, max_values


def load_data(file_path, sequence_length):
    raw_data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=range(8), encoding='utf-8')
    raw_data = raw_data[::-1]  # Reverse for time order
    scaled_data, min_vals, max_vals = normalize(raw_data)

    sequences = [scaled_data[i:i + sequence_length] for i in range(len(scaled_data) - sequence_length)]
    random_idx = np.random.permutation(len(sequences))
    return [sequences[i] for i in random_idx], min_vals, max_vals

def generate_batch(data, batch_size):
    batch_indices = np.random.permutation(len(data))[:batch_size]
    return [data[i] for i in batch_indices], [len(data[i]) for i in batch_indices]

def random_sequence(batch_size, noise_dim, batch_times, max_seq_len):
    return [np.concatenate(
        [np.random.uniform(0, 1, (batch_times[i], noise_dim)), np.zeros((max_seq_len - batch_times[i], noise_dim))],
        axis=0) for i in range(batch_size)]

def custom_rnn(cell_type, hidden_units):
    cell_dict = {
        'gru': tf.nn.rnn_cell.GRUCell,
        'lstm': tf.nn.rnn_cell.BasicLSTMCell
    }
    return cell_dict[cell_type](num_units=hidden_units, activation=tf.nn.tanh)

def time_series_gan(input_data, seq_length, hidden_units=24, rnn_layers=3, training_steps=1, batch_size=128,
                    cell_type='gru', noise_dim=10, adversarial_loss_weight=1):
    max_seq_len = seq_length
    data_dim = len(input_data[0][0])

    data_input = tf.placeholder(tf.float32, [None, max_seq_len, data_dim], name="data_input")
    noise_input = tf.placeholder(tf.float32, [None, max_seq_len, noise_dim], name="noise_input")
    seq_length_input = tf.placeholder(tf.int32, [None], name="seq_length_input")

    def encoder(data_input, seq_length_input):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            rnn_forward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_backward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_forward, rnn_backward, data_input,
                                                             sequence_length=seq_length_input, dtype=tf.float32)
            hidden_states = tf.layers.dense(tf.concat(rnn_outputs, axis=2), hidden_units, activation=tf.nn.sigmoid)
        return hidden_states

    def decodernet(hidden_states, seq_length_input):
        with tf.variable_scope("decodernet", reuse=tf.AUTO_REUSE):
            rnn_forward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_backward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_forward, rnn_backward, hidden_states,
                                                             sequence_length=seq_length_input, dtype=tf.float32)
            reconstructed_data = tf.layers.dense(tf.concat(rnn_outputs, axis=2), data_dim, activation=tf.nn.sigmoid)
        return reconstructed_data

    def generator_net(noise_input, seq_length_input):
        with tf.variable_scope("generator_net", reuse=tf.AUTO_REUSE):
            rnn_forward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_backward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_forward, rnn_backward, noise_input,
                                                             sequence_length=seq_length_input, dtype=tf.float32)
            generated_states = tf.layers.dense(tf.concat(rnn_outputs, axis=2), hidden_units, activation=tf.nn.sigmoid)
        return generated_states

    def supervisor_net(hidden_states, seq_length_input):
        with tf.variable_scope("supervisor_net", reuse=tf.AUTO_REUSE):
            rnn_forward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_backward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_forward, rnn_backward, hidden_states,
                                                             sequence_length=seq_length_input, dtype=tf.float32)
            supervised_states = tf.layers.dense(tf.concat(rnn_outputs, axis=2), hidden_units, activation=tf.nn.sigmoid)
        return supervised_states

    def discriminator_net(hidden_states, seq_length_input):
        with tf.variable_scope("discriminator_net", reuse=tf.AUTO_REUSE):
            rnn_forward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_backward = tf.nn.rnn_cell.MultiRNNCell([custom_rnn(cell_type, hidden_units) for _ in range(rnn_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_forward, rnn_backward, hidden_states,
                                                             sequence_length=seq_length_input, dtype=tf.float32)
            discrim_output = tf.layers.dense(tf.concat(rnn_outputs, axis=2), 1, activation=None)
        return discrim_output

    hidden_states = encoder(data_input, seq_length_input)
    reconstructed_data = decodernet(hidden_states, seq_length_input)
    generated_states = generator_net(noise_input, seq_length_input)
    supervised_generated_states = supervisor_net(generated_states, seq_length_input)
    supervised_real_states = supervisor_net(hidden_states, seq_length_input)
    synthetic_data = decodernet(supervised_generated_states, seq_length_input)

    loss_supervised = tf.losses.mean_squared_error(hidden_states[:, 1:, :], supervised_real_states[:, :-1, :])
    loss_reconstruction = tf.losses.mean_squared_error(data_input, reconstructed_data)
    discriminator_real_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_net(hidden_states, seq_length_input)),
        discriminator_net(hidden_states, seq_length_input))
    discriminator_fake_loss = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_net(supervised_generated_states, seq_length_input)),
        discriminator_net(supervised_generated_states, seq_length_input))
    discriminator_loss = discriminator_real_loss + adversarial_loss_weight * discriminator_fake_loss
    generator_loss = loss_supervised + discriminator_fake_loss

    embed_optimizer = tf.train.AdamOptimizer().minimize(loss_reconstruction)
    discrim_optimizer = tf.train.AdamOptimizer().minimize(discriminator_loss)
    gen_optimizer = tf.train.AdamOptimizer().minimize(generator_loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for step in range(training_steps):
        X_batch, T_batch = generate_batch(input_data, batch_size)
        Z_batch = random_sequence(batch_size, noise_dim, T_batch, max_seq_len)

        session.run(embed_optimizer, feed_dict={data_input: X_batch, seq_length_input: T_batch})
        # Train discriminator and generator
        session.run(discrim_optimizer, feed_dict={data_input: X_batch, seq_length_input: T_batch, noise_input: Z_batch})
        session.run(gen_optimizer, feed_dict={data_input: X_batch, seq_length_input: T_batch, noise_input: Z_batch})

        if step % 1000 == 0:
            d_loss, g_loss = session.run([discriminator_loss, generator_loss],
                                         feed_dict={data_input: X_batch, seq_length_input: T_batch,
                                                    noise_input: Z_batch})
            print(f"Step {step}: Discriminator Loss = {d_loss}, Generator Loss = {g_loss}")

    Z_batch = random_sequence(len(input_data), noise_dim, [seq_length] * len(input_data), max_seq_len)
    generated_sequences = session.run(synthetic_data, feed_dict={noise_input: Z_batch,
                                                                 seq_length_input: [seq_length] * len(input_data)})
    return generated_sequences


def main():
    sequence_length = 24
    file_path = "第二站点补充.csv"
    input_data, min_vals, max_vals = load_data(file_path, sequence_length)
    synthetic_data = time_series_gan(input_data, sequence_length)
    print("Synthetic Data Generated")


if __name__ == "__main__":
    main()
