import tensorflow as tf
from math import sqrt

class SiameseNN(object):
    def __init__(self,
                 embeddings,
                 embeddings_size,
                 embeddings_number,
                 embeddings_pretrain,
                 embeddings_trainable,
                 embeddings_pos_size,
                 embeddings_pos_number,
                 embeddings_pos_trainable,
                 hidden_size,
                 tag_feature,
                 anaphor_feature,
                 ctx_feature,
                 hidden_size_ffl1,
                 hidden_size_ffl2,
                 reg_coef,
                 shortcut,
                 use_ff1,
                 use_ff2):
        """
        SiameseNN model:
            - paper: tbd
            - architecture: tbd
            - training objective: tbd

        Notation:
            - AnaphS: anaphoric sentence = sentence w/ the anaphor
            - positives/negatives: positive/negative candidates

        :param embeddings: pretrained embeddings; numpy array with shape(embeddings_number, embeddings_size)
        :param embeddings_size: the dimension of word embeddings
        :param embeddings_number: the vocabulary size
        :param embeddings_pretrain: use pretrained embeddings
        :param embeddings_trainable: tune embeddings or keep them fixed
        :param embeddings_pos_size: the dimension of TAG embeddings
        :param embeddings_pos_number: the number of TAGS from Stanford Parser
        :param embeddings_pos_trainable: tune TAG embeddings or keep them fixed
        :param hidden_size: the dimension of LSTM hidden state
        :param tag_feature: concat (or not) the TAG embedding to the word embedding (input)
        :param anaphor_feature: concat (or not) the anaphor embedding to the word embedding (input)
        :param hidden_size_ffl1: size of the first feedforward layer of ReLU units
        :param hidden_size_ffl2: size of the second feedforward layer of ReLU units
        :param reg_coef: the L2-regularization rate
        :param shortcut: concate the TAG embedding to representation produced by bi-LSTM
        :param use_ff1: use (or don't) the first feedforward layer of ReLU units
        :param use_ff2: use (or don't) the second feedforward layer of ReLU units
        """

        with tf.name_scope('placeholders'):
            self.sent_pa = tf.placeholder(dtype=tf.int32, shape=[None, None], name="anaph_sent") # AnaphS
            self.sent_pa_len = tf.placeholder(dtype=tf.int32, shape=[None], name="anaphs_len") # length of AnaphS

            self.positive_candidates = tf.placeholder(dtype=tf.int32, shape=[None, None, None],
                                                      name="positive_candidates")
            self.positive_candidates_len = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                          name="positive_candidates_ln") # length of positives
            self.negative_candidates = tf.placeholder(dtype=tf.int32, shape=[None, None, None],
                                                      name="negative_candidates")
            self.negative_candidates_len = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                          name="negative_candidates_len") # length of negatives

            self.anaphors = tf.placeholder(dtype=tf.int32, shape=[None], name="anaphors")

            self.sent_pa_tag = tf.placeholder(dtype=tf.int32, shape=[None, None], name="anaphs_tag") # constituent tag of AnaphS
            self.positive_candidates_tag = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                          name="positive_candidates_tag") # tag of positives
            self.negative_candidates_tag = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                          name="negative_candidates_tag") # tag of negatives
            self.num_positives = tf.placeholder(dtype=tf.int32, shape=[None],
                                                name="real_num_positives") # real number of positives
            self.num_negatives = tf.placeholder(dtype=tf.int32, shape=[None],
                                                name="real_num_negatives") # real number of negatives
            self.ctx = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ctx") # ctx of the anaphor
            self.ctx_len = tf.placeholder(dtype=tf.int32, shape=[None], name="ctx_len") # ctx of the anaphor

            self.keep_rate_input = tf.placeholder(dtype=tf.float32, name="keep_rate_input")
            self.keep_rate_cell_output = tf.placeholder(dtype=tf.float32, name="keep_rate_cell_output")
            self.keep_ffl1_rate = tf.placeholder(dtype=tf.float32, name="self.keep_ffl1_rate")
            self.keep_ffl2_rate = tf.placeholder(dtype=tf.float32, name="self.keep_ffl2_rate")

        with tf.variable_scope("embeddings_lookup"):
            if embeddings_pretrain == "True":
                self.embeddings_tuned = tf.get_variable("pretrained_emb",
                                                        shape=[embeddings_number, embeddings_size],
                                                        initializer=tf.constant_initializer(embeddings),
                                                        trainable=embeddings_trainable,
                                                        dtype=tf.float32)
            else:
                self.embeddings_tuned = tf.get_variable("random_emb",
                                                        shape=[embeddings_number, embeddings_size],
                                                        initializer=tf.random_uniform_initializer(-1, 1, seed=24,
                                                                                                  dtype=tf.float32),
                                                        trainable=embeddings_trainable,
                                                        dtype=tf.float32
                                                        )

            embedded_sent_pa = tf.nn.embedding_lookup(self.embeddings_tuned, self.sent_pa)

            # reshape: pos_candidates_reshape \in [batch_size * num_of_pos_candidates, max_sent_len]
            pos_candidates_reshape = tf.reshape(self.positive_candidates,
                                                shape=[tf.multiply(tf.shape(self.positive_candidates)[0],
                                                                   tf.shape(self.positive_candidates)[1]),
                                                       tf.shape(self.positive_candidates)[2]])

            positive_candidates_len_reshape = tf.reshape(self.positive_candidates_len,
                                                shape=[tf.multiply(tf.shape(self.positive_candidates_len)[0],
                                                                   tf.shape(self.positive_candidates_len)[1])])

            embedded_pos_candidates = tf.nn.embedding_lookup(self.embeddings_tuned, pos_candidates_reshape)

            # reshape: neg_candidates_reshape \in [batch_size * num_of_neg_candidates, max_sent_len]
            neg_candidates_reshape = tf.reshape(self.negative_candidates,
                                                shape=[tf.multiply(tf.shape(self.negative_candidates)[0],
                                                                   tf.shape(self.negative_candidates)[1]),
                                                       tf.shape(self.negative_candidates)[2]])

            negative_candidates_len_reshape = tf.reshape(self.negative_candidates_len,
                                                shape=[tf.multiply(tf.shape(self.negative_candidates_len)[0],
                                                                   tf.shape(self.negative_candidates_len)[1])])

            embedded_neg_candidates = tf.nn.embedding_lookup(self.embeddings_tuned, neg_candidates_reshape)

            if anaphor_feature == "True":
                # anaphor emb. to anaphoric sentences
                anaphors_sent_pa_copy = tf.expand_dims(self.anaphors, 1)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1]])
                anaphors_sent_pa_copy = tf.tile(anaphors_sent_pa_copy, pattern)
                embedded_anaphors_sent_pa = tf.nn.embedding_lookup(self.embeddings_tuned, anaphors_sent_pa_copy)

                # anaphor to positives
                anaphors_sent_pos_copy = tf.expand_dims(self.anaphors, 1)
                pattern = tf.pack([tf.shape(self.positive_candidates)[1], 1])
                anaphors_sent_pos_copy = tf.tile(anaphors_sent_pos_copy, pattern)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1]])
                anaphors_sent_pos_copy = tf.tile(anaphors_sent_pos_copy, pattern)
                embedded_anaphors_pos = tf.nn.embedding_lookup(self.embeddings_tuned, anaphors_sent_pos_copy)

                # anaphor to negatives
                anaphors_sent_neg_copy = tf.expand_dims(self.anaphors, 1)
                pattern = tf.pack([tf.shape(self.negative_candidates)[1], 1])
                anaphors_sent_neg_copy = tf.tile(anaphors_sent_neg_copy, pattern)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1]])
                anaphors_sent_neg_copy = tf.tile(anaphors_sent_neg_copy, pattern)
                embedded_anaphors_neg = tf.nn.embedding_lookup(self.embeddings_tuned, anaphors_sent_neg_copy)

                embedded_sent_pa = tf.concat(2, [embedded_sent_pa, embedded_anaphors_sent_pa])
                embedded_pos_candidates = tf.concat(2, [embedded_pos_candidates, embedded_anaphors_pos])
                embedded_neg_candidates = tf.concat(2, [embedded_neg_candidates, embedded_anaphors_neg])

            if ctx_feature == "True":
                # ctx emb. to to anaphoric sentences
                embedded_ctx = tf.nn.embedding_lookup(self.embeddings_tuned, self.ctx)
                embedded_ctx_sum = tf.reduce_sum(embedded_ctx, reduction_indices=-2)
                embedded_ctx_mean = embedded_ctx_sum / tf.cast(tf.expand_dims(self.ctx_len, -1), dtype=tf.float32)
                embedded_ctx_mean_copy = tf.expand_dims(embedded_ctx_mean, 1)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1], 1])
                embedded_ctx_mean_copy = tf.tile(embedded_ctx_mean_copy, pattern)

                # ctx to positives
                ctx_len_copy = tf.expand_dims(self.ctx_len, 1)
                pattern = tf.pack([tf.shape(self.positive_candidates)[1], 1])
                ctx_pos_len = tf.tile(ctx_len_copy, pattern)
                ctx_pos_len_reshape = tf.reshape(ctx_pos_len,
                                                 [tf.shape(self.sent_pa)[0]*tf.shape(self.positive_candidates)[1]])

                ctx_pos = tf.expand_dims(self.ctx, 1)
                pattern = tf.pack([1, tf.shape(self.positive_candidates)[1], 1])
                ctx_pos_copy = tf.tile(ctx_pos, pattern)
                ctx_pos_copy_reshape = tf.reshape(ctx_pos_copy,
                                                  [tf.shape(self.sent_pa)[0]*tf.shape(self.positive_candidates)[1],
                                                   tf.shape(self.ctx)[1]]
                                                  )
                embedded_ctx_pos = tf.nn.embedding_lookup(self.embeddings_tuned, ctx_pos_copy_reshape)
                embedded_ctx_pos_sum = tf.reduce_sum(embedded_ctx_pos, reduction_indices=-2)
                embedded_ctx_pos_mean = embedded_ctx_pos_sum / tf.cast(tf.expand_dims(ctx_pos_len_reshape, -1), dtype=tf.float32)
                embedded_ctx_pos_mean_copy = tf.expand_dims(embedded_ctx_pos_mean, 1)
                pattern = tf.pack([1,tf.shape(self.sent_pa)[1], 1])
                embedded_ctx_pos_mean_copy = tf.tile(embedded_ctx_pos_mean_copy, pattern)
                self.embedded_ctx_pos_mean_copy_reshape = tf.reshape(embedded_ctx_pos_mean_copy,
                                                                [tf.shape(self.sent_pa)[0]*tf.shape(self.positive_candidates)[1],
                                                                 tf.shape(self.sent_pa)[1],
                                                                 embeddings_size])
                # ctx to negatives
                ctx_neg = tf.expand_dims(self.ctx, 1)
                pattern = tf.pack([1, tf.shape(self.negative_candidates)[1], 1])
                ctx_neg_copy = tf.tile(ctx_neg, pattern)

                ctx_len_copy = tf.expand_dims(self.ctx_len, 1)
                pattern = tf.pack([tf.shape(self.negative_candidates)[1], 1])
                ctx_neg_len = tf.tile(ctx_len_copy, pattern)
                ctx_neg_len_reshape = tf.reshape(ctx_neg_len,
                                                 [tf.shape(self.sent_pa)[0] * tf.shape(self.negative_candidates)[1]])

                ctx_neg_copy_reshape = tf.reshape(ctx_neg_copy,
                                                  [tf.shape(self.sent_pa)[0] * tf.shape(self.negative_candidates)[1],
                                                   tf.shape(self.ctx)[1]]
                                                  )
                embedded_ctx_neg = tf.nn.embedding_lookup(self.embeddings_tuned, ctx_neg_copy_reshape)
                embedded_ctx_neg_sum = tf.reduce_sum(embedded_ctx_neg, reduction_indices=-2)
                embedded_ctx_neg_mean = embedded_ctx_neg_sum / tf.cast(tf.expand_dims(ctx_neg_len_reshape, -1),
                                                                       dtype=tf.float32)
                embedded_ctx_neg_mean_copy = tf.expand_dims(embedded_ctx_neg_mean, 1)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1], 1])
                embedded_ctx_neg_mean_copy = tf.tile(embedded_ctx_neg_mean_copy, pattern)
                embedded_ctx_neg_mean_copy_reshape = tf.reshape(embedded_ctx_neg_mean_copy,
                                                                [tf.shape(self.sent_pa)[0]*tf.shape(self.negative_candidates)[1],
                                                                 tf.shape(self.sent_pa)[1],
                                                                 embeddings_size])

                embedded_sent_pa = tf.concat(2, [embedded_sent_pa, embedded_ctx_mean_copy])
                embedded_pos_candidates = tf.concat(2, [embedded_pos_candidates, self.embedded_ctx_pos_mean_copy_reshape])
                embedded_neg_candidates = tf.concat(2, [embedded_neg_candidates, embedded_ctx_neg_mean_copy_reshape])

            if tag_feature == "True":
                uniform_param = sqrt(1.0 / float(embeddings_pos_number + embeddings_pos_size))
                self.embeddings_pos = tf.get_variable("pos_emb",
                                                      shape=[embeddings_pos_number, embeddings_pos_size],
                                                      initializer=tf.random_uniform_initializer(-uniform_param, uniform_param, seed=24, dtype=tf.float32),
                                                      trainable=embeddings_pos_trainable,
                                                      dtype=tf.float32)

                embedded_sent_pa_tag = tf.nn.embedding_lookup(self.embeddings_pos, self.sent_pa_tag)

                pos_tag = tf.nn.embedding_lookup(self.embeddings_pos, self.positive_candidates_tag)
                pos_tag_reshape = tf.reshape(pos_tag, [tf.multiply(tf.shape(self.sent_pa)[0],
                                                                   tf.shape(self.positive_candidates)[1]),
                                                       embeddings_pos_size])
                pos_tag_copy = tf.expand_dims(pos_tag_reshape, 1)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1], 1])
                embedded_pos_tag = tf.tile(pos_tag_copy, pattern)

                neg_tag = tf.nn.embedding_lookup(self.embeddings_pos, self.negative_candidates_tag)
                neg_tag_reshape = tf.reshape(neg_tag, [tf.multiply(tf.shape(self.sent_pa)[0],
                                                                   tf.shape(self.negative_candidates)[1]),
                                                       embeddings_pos_size])
                neg_tag_copy = tf.expand_dims(neg_tag_reshape, 1)
                pattern = tf.pack([1, tf.shape(self.sent_pa)[1], 1])
                embedded_neg_tag = tf.tile(neg_tag_copy, pattern)

                embedded_sent_pa = tf.concat(2, [embedded_sent_pa, embedded_sent_pa_tag])
                embedded_pos_candidates = tf.concat(2, [embedded_pos_candidates, embedded_pos_tag])
                embedded_neg_candidates = tf.concat(2, [embedded_neg_candidates, embedded_neg_tag])

            embedded_sent_pa = tf.nn.dropout(embedded_sent_pa, keep_prob=self.keep_rate_input)
            embedded_pos_candidates = tf.nn.dropout(embedded_pos_candidates, keep_prob=self.keep_rate_input)
            embedded_neg_candidates = tf.nn.dropout(embedded_neg_candidates, keep_prob = self.keep_rate_input)

        with tf.name_scope("siamese"):
            with tf.variable_scope("ffl1") as ffl1_scope:
                if shortcut == "True":
                    weights_size = 2 * hidden_size + embeddings_pos_size
                else:
                    weights_size = 2 * hidden_size
                weights = tf.get_variable("weights_ffl1",
                                          shape=[weights_size, hidden_size_ffl1],
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=24, dtype=tf.float32),
                                          trainable=True,
                                          dtype=tf.float32)

                biases = tf.get_variable("biases_ffl1",
                                         shape=[hidden_size_ffl1],
                                         initializer=tf.constant_initializer(value=0),
                                         trainable=True,
                                         dtype=tf.float32)

            with tf.variable_scope("score_layer") as score_layer_scope:
                if use_ff2 == "True":
                    weights_size = hidden_size_ffl2
                else:
                    if use_ff1 == "True":
                        weights_size = 2 * hidden_size_ffl1
                    else:
                        if shortcut == "True":
                            weights_size = 2 * (2 * hidden_size + embeddings_pos_size)
                        else:
                            weights_size = 4 * hidden_size

                weights_score = tf.get_variable("weights_score",
                                          shape = [weights_size, 1],
                                                initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                           mode='FAN_IN',
                                                                                                           uniform=False,
                                                                                                           seed=24,
                                                                                                           dtype=tf.float32),
                                                trainable=True,
                                          dtype=tf.float32)

                biases_score = tf.get_variable("biases_score",
                                         shape=[1],
                                         initializer=tf.constant_initializer(value=0),
                                         trainable=True,
                                         dtype=tf.float32)

            with tf.variable_scope("ffl2") as ffl2_scope:
                if use_ff1 == "True":
                    weights_size_1 = 2 * hidden_size_ffl1
                if use_ff1 == "False":
                    if shortcut == "True":
                        weights_size_1 = 2 * (2*hidden_size + embeddings_pos_size)
                    else:
                        weights_size_1 = 4 * hidden_size

                weights_joint = tf.get_variable("weights_ffl2",
                                                  shape = [weights_size_1, hidden_size_ffl2],
                                                initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                           mode='FAN_IN',
                                                                                                           uniform=False,
                                                                                                           seed=24,
                                                                                                           dtype=tf.float32),

                                                trainable=True,
                                                  dtype=tf.float32)

                biases_joint = tf.get_variable("biases_ffl2",
                                         shape=[hidden_size_ffl2],
                                         initializer=tf.constant_initializer(value=0),
                                         trainable=True,
                                         dtype=tf.float32)

            with tf.variable_scope("LSTM") as rnn_scope:
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                  state_is_tuple=True,
                                                  initializer=tf.orthogonal_initializer(seed=24))

                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                  state_is_tuple=True,
                                                  initializer=tf.orthogonal_initializer(seed=24))

                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_rate_cell_output)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_rate_cell_output)

                outputs_sent_pa, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                     cell_bw=cell_bw,
                                                                     dtype=tf.float32,
                                                                     sequence_length=self.sent_pa_len,
                                                                     inputs=embedded_sent_pa)
                output_fw, output_bw = outputs_sent_pa

                outputs = [output_fw, output_bw]
                outputs_sent_pa_concat = tf.concat(2, outputs, name='output_states_anaphs')
                outputs_sent_pa_concat_aggregated = tf.reduce_sum(outputs_sent_pa_concat, reduction_indices=-2)
                outputs_sent_pa_mean = outputs_sent_pa_concat_aggregated / tf.cast(tf.expand_dims(self.sent_pa_len, -1),
                                                                                   dtype=tf.float32)
                outputs_sent_pa_mean_reshape = tf.reshape(outputs_sent_pa_mean,
                                                          shape=[tf.shape(self.sent_pa)[0],
                                                                 2 * hidden_size])
                if shortcut == "True":
                    outputs_sent_pa_mean_reshape = tf.concat(1, [outputs_sent_pa_mean_reshape,
                                                                 tf.reduce_mean(embedded_sent_pa_tag, reduction_indices=-2)])
                if use_ff1 == "True":
                    final_states_sent_pa = tf.nn.elu(tf.add(tf.matmul(outputs_sent_pa_mean_reshape, weights), biases), name="anaphs_elu")
                    final_states_sent_pa = tf.nn.dropout(final_states_sent_pa, keep_prob=self.keep_ffl1_rate)

                if use_ff1 == "False":
                    final_states_sent_pa = outputs_sent_pa_mean_reshape

                rnn_scope.reuse_variables()
                #gru_scope.reuse_variables()
                outputs_pos_candidates, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            dtype=tf.float32,
                                                                            sequence_length=positive_candidates_len_reshape,
                                                                            inputs=embedded_pos_candidates
                                                                            )
                output_fw_pos, output_bw_pos = outputs_pos_candidates

                outputs_pos = [output_fw_pos, output_bw_pos]
                outputs_pos_candidates_concat = tf.concat(2, outputs_pos)

                outputs_pos_candidates_concat_aggregated = tf.reduce_sum(outputs_pos_candidates_concat, reduction_indices=-2)

                outputs_pos_candidates_mean = outputs_pos_candidates_concat_aggregated / tf.cast(tf.expand_dims(positive_candidates_len_reshape, -1), dtype=tf.float32)

                outputs_pos_candidates_mean_reshape = tf.reshape(outputs_pos_candidates_mean,
                                                                 shape=[tf.shape(pos_candidates_reshape)[0],
                                                                        2 * hidden_size])
                if shortcut == "True":
                    outputs_pos_candidates_mean_reshape = tf.concat(1, [outputs_pos_candidates_mean_reshape, pos_tag_reshape])

                if use_ff1 == "True":
                    final_states_pos_candidates = tf.nn.elu(tf.add(tf.matmul(outputs_pos_candidates_mean_reshape, weights), biases), name="positives_elu")
                    final_states_pos_candidates = tf.nn.dropout(final_states_pos_candidates, keep_prob=self.keep_ffl1_rate)
                if use_ff1 == "False":
                    final_states_pos_candidates = outputs_pos_candidates_mean_reshape

                rnn_scope.reuse_variables()
                #gru_scope.reuse_variables()
                outputs_neg_candidates, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            dtype=tf.float32,
                                                                            sequence_length=negative_candidates_len_reshape,
                                                                            inputs=embedded_neg_candidates
                                                                  )
                output_fw_neg, output_bw_neg = outputs_neg_candidates

                outputs_neg = [output_fw_neg, output_bw_neg]
                outputs_neg_candidates_concat = tf.concat(2, outputs_neg)
                outputs_neg_candidates_concat_aggregated = tf.reduce_sum(outputs_neg_candidates_concat, reduction_indices=-2)
                outputs_neg_candidates_mean = outputs_neg_candidates_concat_aggregated / tf.cast(tf.expand_dims(negative_candidates_len_reshape, -1), dtype=tf.float32)
                outputs_neg_candidates_mean_reshape = tf.reshape(outputs_neg_candidates_mean,
                                                                              shape=[
                                                                                  tf.shape(neg_candidates_reshape)[0],
                                                                                  2 * hidden_size])
                if shortcut == "True":
                    outputs_neg_candidates_mean_reshape = tf.concat(1, [outputs_neg_candidates_mean_reshape, neg_tag_reshape])
                    hidden_size = 2 * hidden_size + embeddings_pos_size

                if use_ff1 == "True":
                    final_states_neg_candidates = tf.nn.elu(tf.add(tf.matmul(outputs_neg_candidates_mean_reshape, weights), biases), name="negatives_elu")
                    final_states_neg_candidates = tf.nn.dropout(final_states_neg_candidates, keep_prob=self.keep_ffl1_rate)
                    hidden_size = hidden_size_ffl1

                if use_ff1 == "False":
                    final_states_neg_candidates = outputs_neg_candidates_mean_reshape

            with tf.name_scope("scores"):
                # loss = the slack-rescaled max-margin training objective
                # take a look at: http://arxiv.org/abs/1606.01323

                # copy every row of final_states_sent_pa (num_of_pos_candidates) times
                final_states_sent_pa_copy = tf.expand_dims(final_states_sent_pa, 1)
                pattern = tf.pack([1, tf.shape(self.positive_candidates)[1], 1])
                final_states_sent_pa_copy = tf.tile(final_states_sent_pa_copy, pattern)
                final_states_sent_pa_copy_reshape = tf.reshape(final_states_sent_pa_copy,
                                                            [tf.multiply(tf.shape(self.positive_candidates)[1],
                                                                        tf.shape(self.sent_pa)[0]),
                                                             hidden_size])
                # final_states_sent_pa_copy_reshape \in [batch_size * num_of_pos_candidates, hidden_size]

                joint_representation_pos = tf.concat(1, [tf.abs(tf.subtract(final_states_pos_candidates, final_states_sent_pa_copy_reshape)),
                                                     tf.multiply(final_states_pos_candidates, final_states_sent_pa_copy_reshape)])

                if use_ff2 == "True":
                    joint_representation_pos = tf.nn.elu(tf.add(tf.matmul(joint_representation_pos, weights_joint), biases_joint), name="joint_positives_elu")
                    joint_representation_pos = tf.nn.dropout(joint_representation_pos, keep_prob=self.keep_ffl2_rate)

                scores_pos_candidates = tf.add(tf.matmul(joint_representation_pos, weights_score), biases_score)

                # scores_positive_candidates \in [batch_size * num_of_positive_candidates]

                scores_pos_candidates_reshape = tf.reshape(scores_pos_candidates,
                                                           tf.shape(self.positive_candidates)[:2])
                # after reshape \in [batch_size, num_of_positive_candidates]

                positives_mask = tf.cast(tf.sequence_mask(self.num_positives,
                                                          tf.shape(self.positive_candidates)[1]),
                                         dtype=tf.float32)
                #positives_mask \in [batch, max_positives_num]

                max_score_pos_candidates = tf.reduce_max(tf.multiply(scores_pos_candidates_reshape, positives_mask), 1)
                # max_score_positive_candidates \in [batch_size]


                # copy every row of max_score_positive_candidates (num_of_neg_candidates) times
                max_score_pos_candidates_copy = tf.expand_dims(max_score_pos_candidates, 1)
                pattern = tf.pack([1, tf.shape(self.negative_candidates)[1]])
                max_score_pos_candidates_copy = tf.tile(max_score_pos_candidates_copy, pattern)
                max_score_pos_candidates_copy_reshape = tf.reshape(max_score_pos_candidates_copy,
                                                                   [tf.multiply(tf.shape(self.negative_candidates)[1],
                                                                        tf.shape(self.sent_pa)[0]), 1])

                # max_score_pos_candidates_copy_reshape \in [batch_size * num_of_neg_candidates]

                # copy every row of final_states_sent_pa (num_of_neg_candidates) times
                final_states_sent_pa_new_copy = tf.expand_dims(final_states_sent_pa, 1)
                pattern = tf.pack([1, tf.shape(self.negative_candidates)[1], 1])
                final_states_sent_pa_new_copy = tf.tile(final_states_sent_pa_new_copy, pattern)
                final_states_sent_pa_copy_new_reshape = tf.reshape(final_states_sent_pa_new_copy,
                                                            [tf.multiply(tf.shape(self.negative_candidates)[1],
                                                                        tf.shape(self.sent_pa)[0]),
                                                             hidden_size])
                # final_states_sent_pa_copy_new_reshape \in [batch_size * num_of_neg_candidates, hidden_size]

                # NOTE: tf.add supports broadcasting
                joint_representation_neg = tf.concat(1, [tf.abs(tf.subtract(final_states_neg_candidates,
                                                                 final_states_sent_pa_copy_new_reshape)),
                                                          tf.multiply(final_states_neg_candidates,
                                                                 final_states_sent_pa_copy_new_reshape)])
                if use_ff2 == "True":
                    joint_representation_neg = tf.nn.elu(tf.add(tf.matmul(joint_representation_neg, weights_joint), biases_joint), name="joint_negatives_elu")
                    joint_representation_neg = tf.nn.dropout(joint_representation_neg, keep_prob=self.keep_ffl2_rate)

                scores_neg_candidates = tf.add(tf.matmul(joint_representation_neg, weights_score), biases_score)

                # scores_neg_candidates \in [batch_size * num_of_negative_candidates]

                scores_neg_candidates_reshape = tf.reshape(scores_neg_candidates,
                                                           tf.shape(self.negative_candidates)[:2])
                # scores_neg_candidates_reshape \in [batch, num_of_negative_candidates]

                self.scores = tf.concat(1, [scores_pos_candidates_reshape, scores_neg_candidates_reshape], name='scores')
                # self.scores \in [batch, num_pos_candidates + num_negative_examples]
                self.scores_argmax = tf.argmax(self.scores, 1, name='scores_argmax')
                # self.scores_argmax \in [num_pos_candidates + num_negative_examples]

            with tf.name_scope("loss"):
                loss_temp = tf.add(tf.subtract(scores_neg_candidates, max_score_pos_candidates_copy_reshape), tf.to_float(1))
                # loss_temp \in [batch * num_of_neg_candidates]

                loss_temp_reshape = tf.reshape(loss_temp, tf.shape(self.negative_candidates)[:2])
                # loss_temp_reshape \in [batch, num_of_neg_candidates]

                negatives_mask = tf.cast(tf.sequence_mask(self.num_negatives,
                                                          tf.shape(self.negative_candidates)[1]),
                                         dtype=tf.float32)
                loss_max = tf.maximum(tf.reduce_max(tf.multiply(loss_temp_reshape, negatives_mask), 1), 0)

                tv = tf.trainable_variables()
                regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
                self.loss = tf.reduce_sum(loss_max) + reg_coef * regularization_cost