from models.biLSTM_SiameseMR import SiameseNN
from operator import add
from scipy.stats import rankdata
import tensorflow as tf
import time as ti
import os
import numpy as np
import logging
import matplotlib
from matplotlib import style
style.use('seaborn-whitegrid')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(argv):
    train_batches = argv.train_batches
    dev_batches = argv.dev_batches
    test_batches = argv.test_batches
    nominal_test_batches = argv.nominal_test_batches
    pronominal_test_batches = argv.pronominal_test_batches
    embeddings = argv.embeddings
    pos_vocabulary = argv.pos_vocabulary

    tf.reset_default_graph()

    with tf.Graph().as_default():
        #tf.set_random_seed(24)
        gpu_options = tf.GPUOptions(allow_growth=True)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True,
                                      gpu_options=gpu_options)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            pa_model = SiameseNN(embeddings=embeddings,
                                 embeddings_size=embeddings.shape[1],
                                 embeddings_number=embeddings.shape[0],
                                 embeddings_pretrain=argv.pretrained_emb,
                                 embeddings_trainable=argv.train_emb,
                                 embeddings_pos_size=argv.pos_emb_size,
                                 embeddings_pos_number=len(pos_vocabulary),
                                 embeddings_pos_trainable=argv.train_pos_emb,
                                 hidden_size=argv.hidden_size,
                                 tag_feature=argv.tag_feature,
                                 anaphor_feature=argv.anaphor_feature,
                                 ctx_feature=argv.ctx_feature,
                                 hidden_size_ffl1 = argv.hidden_size_ffl1,
                                 hidden_size_ffl2 = argv.hidden_size_ffl2,
                                 reg_coef=argv.reg_coef,
                                 shortcut=argv.shortcut,
                                 use_ff1=argv.use_ff1,
                                 use_ff2=argv.use_ff2
                                 )

            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            logging.info('Total_params: %d\n' % param_stats.total_parameters)

            if argv.opt == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=argv.lr)

            if argv.opt == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=argv.lr)

            if argv.opt == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=argv.lr)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            params = tf.trainable_variables()
            gradients = tf.gradients(pa_model.loss, params)

            clipped_gradients, norm = tf.clip_by_global_norm(gradients, argv.grad_clip)
            gradient_norms = norm
            updates = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

            ##### UNCOMMENT IF YOU WANT TO USE SUMMARIES###
            '''
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimize = tf.contrib.layers.optimize_loss(pa_model.loss,
                                                        global_step=global_step,
                                                        learning_rate=argv.lr,
                                                        optimizer=optimizer,
                                                        clip_gradients=argv.grad_clip)


            # Keep track of gradient values and sparsity (optional)
            grads_and_vars = optimizer.compute_gradients(pa_model.loss)
            for gv in grads_and_vars:
                logging.info(str(gv[0]) + " - " + gv[1].name)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            '''

            # output directory for models
            timestamp = str(int(ti.time()))
            fname = "runs/ranking_" + argv.train_corpus + "_" + argv.candidates_num + '_arch_id_' + argv.arch_id
            out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                                   fname,
                                                   timestamp))
            logging.info("Writing to %s " % out_dir)
            '''
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", pa_model.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            '''

            # checkpoint setup
            checkpoints_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_best = os.path.join(checkpoints_dir, "model")
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            saver = tf.train.Saver(tf.global_variables())

            def step(batch, eval=False):
                anaphors, sent_pa,\
                positive_candidates, negative_candidates,\
                positive_candidates_tag, negative_candidates_tag,\
                sent_anaph_len, positive_candidates_len, negative_candidates_len,\
                sent_pa_tag, num_positives, num_negatives, ctx_all, ctx_len = zip(*batch)

                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                if not eval:
                    keep_rate_input = argv.keep_rate_input
                    keep_rate_cell_output = argv.keep_rate_cell_output
                    keep_ffl1_rate = argv.keep_ffl1_rate
                    keep_ffl2_rate = argv.keep_ffl2_rate
                else:
                    keep_rate_input = 1.0
                    keep_rate_cell_output = 1.0
                    keep_ffl1_rate = 1.0
                    keep_ffl2_rate = 1.0

                feed_dict = {pa_model.sent_pa: np.asarray(sent_pa, dtype=np.int32),
                             pa_model.sent_pa_len: np.asarray(sent_anaph_len, dtype=np.int32),
                             pa_model.positive_candidates: np.asarray(positive_candidates, dtype=np.int32),
                             pa_model.positive_candidates_len: np.asarray(positive_candidates_len, dtype=np.int32),
                             pa_model.negative_candidates: np.asarray(negative_candidates, dtype=np.int32),
                             pa_model.negative_candidates_len: np.asarray(negative_candidates_len, dtype=np.int32),
                             pa_model.anaphors: np.asarray(anaphors, dtype=np.int32),
                             pa_model.positive_candidates_tag: np.asarray(positive_candidates_tag, dtype=np.int32),
                             pa_model.negative_candidates_tag: np.asarray(negative_candidates_tag, dtype=np.int32),
                             pa_model.sent_pa_tag: np.asarray(sent_pa_tag, dtype=np.int32),
                             pa_model.num_positives: np.asarray(num_positives, dtype=np.int32),
                             pa_model.num_negatives: np.asarray(num_negatives, dtype=np.int32),
                             pa_model.ctx: np.asarray(ctx_all, dtype=np.int32),
                             pa_model.ctx_len: np.asarray(ctx_len, dtype=np.int32),
                             pa_model.keep_rate_input: keep_rate_input,
                             pa_model.keep_rate_cell_output: keep_rate_cell_output,
                             pa_model.keep_ffl1_rate: keep_ffl1_rate,
                             pa_model.keep_ffl2_rate: keep_ffl2_rate}

                if not eval:
                    ##### for summaries uncomment lines #####
                    _, _, step, scores, loss = sess.run([updates,
                                                         gradient_norms,
                                                global_step,
                                                #train_summary_op,
                                                pa_model.scores,
                                                pa_model.loss],
                                                feed_dict=feed_dict)
                    #train_summary_writer.add_summary(summaries, step)
                    return scores, loss


                else:
                    scores, loss = sess.run([pa_model.scores,
                                             pa_model.loss],
                                             feed_dict)
                    return scores, loss

        init_vars = tf.global_variables_initializer()
        sess.run(init_vars)

        logging.info("Start training in %s epochs" % argv.num_epoch)

        current_step = 0
        total_time = 0.0
        precisions_train = []
        precisions_test = []
        precisions_nominal_test = []
        precisions_pronominal_test = []
        precisions_dev = []

        best_precision = 0.0
        num_epoch = argv.num_epoch
        best_dev_test = [0.0]*4

        for epoch in range(0, argv.num_epoch):
            logging.info("EPOCH %s" % (epoch + 1))

            start_epoch = ti.time()

            train_loss = 0.0
            ns = 4

            pn_train = [0]*ns
            for train_batch in train_batches:
                scores_train, loss_train = step(train_batch)

                train_loss += loss_train
                _, sent_pa, positive_candidates, negative_candidates, _, _, _, _, _, _, num_positives, _, _, _ = zip(*train_batch)
                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                assert len(scores_train) == len(sent_pa)
                all_candidates = len(positive_candidates[0]) + len(negative_candidates[0])
                assert len(scores_train[0]) == all_candidates

                pn_batch = precision_n(scores_train, num_positives, ns)
                pn_train = map(add, pn_batch, pn_train)
            pn_train[:] = [x / (float(len(train_batches))) for x in pn_train]
            precision_train = pn_train[0]
            precisions_train.append(precision_train)
            train_loss /= (float(len(train_batches)))
            logging.info('train loss: %s' % train_loss)
            logging.info('train precision at %s' % ns)
            logging.info(pn_train)

            pn_dev = [0]*ns
            for dev_batch in dev_batches:
                scores_dev, loss_dev = step(dev_batch, eval=True)

                _, sent_pa, positive_candidates, negative_candidates, _, _, _, _, _, _, num_positives, _, _, _ = zip(*dev_batch)
                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                assert len(scores_dev) == len(sent_pa)
                all_candidates = len(positive_candidates[0]) + len(negative_candidates[0])
                assert len(scores_dev[0]) == all_candidates
                pn_batch = precision_n(scores_dev, num_positives, ns)
                pn_dev = map(add, pn_batch, pn_dev)

            pn_dev[:] = [x / (float(len(dev_batches))) for x in pn_dev]
            precision_dev = pn_dev[0]
            precisions_dev.append(precision_dev)
            logging.info('Dev precision at %s:' % ns)
            logging.info(pn_dev)

            pn_test = [0]*ns
            for test_batch in test_batches:
                scores_test, loss_test = step(test_batch, eval=True)

                _, sent_pa, positive_candidates, negative_candidates, _, _, _, _, _, _, num_positives, _, _, _ = zip(*test_batch)
                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                assert len(scores_test) == len(sent_pa)
                all_candidates = len(positive_candidates[0]) + len(negative_candidates[0])
                assert len(scores_test[0]) == all_candidates
                pn_batch = precision_n(scores_test, num_positives, ns)
                pn_test = map(add, pn_batch, pn_test)

            pn_test[:] = [x / (float(len(test_batches))) for x in pn_test]
            precision_test = pn_test[0]
            precisions_test.append(precision_test)
            logging.info('Test precision at %s:' % ns)
            logging.info(pn_test)

            pn_pronominal_test = [0]*ns
            for test_batch in pronominal_test_batches:
                scores_test, loss_test = step(test_batch, eval=True)

                _, sent_pa, positive_candidates, negative_candidates, _, _, _, _, _, _, num_positives, _, _, _ = zip(*test_batch)
                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                assert len(scores_test) == len(sent_pa)
                all_candidates = len(positive_candidates[0]) + len(negative_candidates[0])
                assert len(scores_test[0]) == all_candidates
                pn_batch = precision_n(scores_test, num_positives, ns)
                pn_pronominal_test = map(add, pn_batch, pn_pronominal_test)
            pn_pronominal_test[:] = [x / (float(len(pronominal_test_batches))) for x in pn_pronominal_test]
            precision_test = pn_pronominal_test[0]
            precisions_pronominal_test.append(precision_test)
            logging.info('pronominal test precision at %s:' % ns)
            logging.info(pn_pronominal_test)

            pn_nominal_test = [0]*ns
            for test_batch in nominal_test_batches:
                scores_test, loss_test = step(test_batch, eval=True)

                _, sent_pa, positive_candidates, negative_candidates, _, _, _, _, _, _, num_positives, _, _, _ = zip(*test_batch)
                positive_candidates = list(positive_candidates)
                negative_candidates = list(negative_candidates)

                assert len(scores_test) == len(sent_pa)
                all_candidates = len(positive_candidates[0]) + len(negative_candidates[0])
                assert len(scores_test[0]) == all_candidates
                pn_batch = precision_n(scores_test, num_positives, ns)
                pn_nominal_test = map(add, pn_batch, pn_nominal_test)
            pn_nominal_test[:] = [x / (float(len(nominal_test_batches))) for x in pn_nominal_test]
            precision_test = pn_nominal_test[0]
            precisions_nominal_test.append(precision_test)
            logging.info('nominal test precision at %s:' % ns)
            logging.info(pn_nominal_test)

            if precision_dev > best_precision:
                logging.info("Better precision!")

                best_precision = precision_dev
                best_dev_test = pn_test
                best_dev_nominal_test = pn_nominal_test
                best_dev_pronominal_test = pn_pronominal_test

                # save
                path = saver.save(sess, checkpoint_best)

                logging.info("Saved best model checkpoint to {}\n".format(path))

            epoch_time = (ti.time() - start_epoch) / float(60)
            logging.info("trained and evaluated epoch %s in time %s minutes" % (epoch + 1, epoch_time))

            total_time += epoch_time
            logging.info("Total time in %s epochs: %s" % (epoch + 1, total_time))

            if precision_train > 99:
                num_epoch = epoch + 1
                break

        logging.info("Saving performance figure...")
        plt.figure(dpi=400)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12
        steps = range(1, num_epoch+1)
        plt.plot(steps, precisions_train, linewidth=2, color='#6699ff', linestyle='-', marker='o', markeredgecolor='black',
         markeredgewidth=0.5, label='train')
        plt.plot(steps, precisions_test, linewidth=6, color='#ff4d4d', linestyle='-', marker='D', markeredgecolor='black',
         markeredgewidth=0.5, label='test')
        plt.plot(steps, precisions_nominal_test, linewidth=6, color='#ff3300', linestyle='-', marker='D',
                 markeredgecolor='black',
                 markeredgewidth=0.5, label='test nominal')
        plt.plot(steps, precisions_pronominal_test, linewidth=6, color='#660033', linestyle='-', marker='D',
                 markeredgecolor='black',
                 markeredgewidth=0.5, label='test pronominal')
        plt.plot(steps, precisions_dev, linewidth=4, color='#ffcc66', linestyle='-', marker='s', markeredgecolor='black',
         markeredgewidth=0.5, label='dev')
        plt.xlabel('epochs')
        plt.ylabel('s @ 1')
        plt.legend(loc='best', numpoints=1, fancybox=True)
        fig_path = "figs/" + argv.train_corpus + '_arch_id_' + argv.arch_id + ".png"
        plt.savefig(fig_path)

        return best_dev_test, best_dev_nominal_test, best_dev_pronominal_test


def precision_n(test_scores, num_true, n):
    """
    Precision at n measure is the number of instances where the any crowd's answer occur within ranker's firs n choices
    For more details take a look at: http://www.aclweb.org/anthology/D13-1030
    The first num_true_antec[i] dev_scores are predicted scores for true antecedents of the i-th sentence w/ PA

    :param test_scores: \in [batch_size, num of candidates], for every sent w\ PA predicted scores for its candidates
    :param num_true_antec: \in [batch_size], for every sent w\ PA number of true antecedents
    :return: list of size 10
    """
    precisions = []
    for i in range(n):
        precision = 0
        for k, item in enumerate(test_scores):
            ranks = len(item) - rankdata(item, method='ordinal').astype(int)
            precision += min(1, len(set(ranks[:num_true[k]]) & set(range(i+1))))
        precision /= float(len(test_scores))
        precision *= 100
        precisions.append(precision)
    return precisions