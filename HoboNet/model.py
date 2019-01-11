import tensorflow as tf


def build_model(num_classes,
                state_size=100,
                embedding_size=256,
                learning_rate=1e-4):

    with tf.varoable_scope('HoboNet'):
        with tf.variable_scope('encoder'):
            batch_size = tf.placeholder(tf.int64, name='batch_size')

            x = tf.placeholder(tf.int32, shape=(None, None), name='input_placeholder')
            y = tf.placeholder(tf.int32, shape=(None, None), name='labels_placeholder')

            # The following allows the model to dynamically accept different sizes of x and also
            # produces batches of the input data.
            dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()

            data_iter = dataset.make_initializable_iterator()
            features, labels = data_iter.get_next()

            # Embeddings convert each integer to a vector representation
            embeddings = tf.get_variable('embedding_matrix', [num_classes, embedding_size])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, features)
            dec_inputs = tf.nn.embedding_lookup(embeddings, labels)

            # Encoder
            fw_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)

            # The bidirectional model reads in the input from beginning to end and from end to begninng
            # and outputs both the forward and back outputs and the forward and backward states.
            dynam_batch_size = tf.shape(features)[0]
            fw_init_state = fw_cell.zero_state(dynam_batch_size, tf.float32)
            bw_init_state = bw_cell.zero_state(dynam_batch_size, tf.float32)
            (fw_output, bw_output), (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                rnn_inputs,
                initial_state_fw=fw_init_state,
                initial_state_bw=bw_init_state)

            # Decoder
            # The forward and backward states need to be decomposed and concatenated into a single state in order to
            # feed it into the LSTM decoder. This results in a state size that is twice as large as the encoder state size.

            enc_final_state_c = tf.concat((fw_final_state.c, bw_final_state.c), 1)
            enc_final_state_h = tf.concat((fw_final_state.h, bw_final_state.h), 1)
            enc_final_state = tf.contrib.rnn.LSTMStateTuple(c=enc_final_state_c,
                                                            h=enc_final_state_h)
        with tf.variable_scope('decoder'):
            W = tf.get_variable('W', [2 * state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

            decoder_cell = tf.nn.rnn_cell.LSTMCell(2 * state_size, state_is_tuple=True)
            decoder_init_state = decoder_cell.zero_state(dynam_batch_size, tf.float32)

            rnn_outputs, final_state = tf.nn.dynamic_rnn(decoder_cell, dec_inputs, initial_state=enc_final_state)

            # reshape rnn_outputs and y so we can get the logits in a single matmul
            rnn_outputs = tf.reshape(rnn_outputs, [-1, 2 * state_size])
            y_reshaped = tf.reshape(labels, [-1])

            logits = tf.matmul(rnn_outputs, W) + b

            # Greedy search
            predictions = tf.argmax(logits, axis=1, name='preds')

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))

    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)

    train_summary = tf.summary.scalar('training_loss', total_loss)

    # The following are accessible once the graph has been built:
    return dict(
        x=x,
        y=y,
        fw_init=fw_init_state,
        bw_init=bw_init_state,
        batch_size=batch_size,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        logits=logits,
        data_iter=data_iter,
        saver=tf.train.Saver(),
        train_summary=train_summary
    )
