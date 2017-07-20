from scipy.stats import rankdata
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
from operator import add
import logging
#import prettyplotlib as ppl
import random
import matplotlib
from matplotlib import style
style.use('seaborn-whitegrid')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs

UNK = "<UNK>"


def plot(candidates_tsne, all_candidates_tag, num_positives, file_path, ranks):
    almost_black = '#262626'

    fig, ax = plt.subplots(1)

    x = [candidates_tsne[0, 0]]
    y = [candidates_tsne[0, 1]]
    ngram_text = ["AnaphS (S)"]
    ax.scatter(x, y, label='AnaphS', alpha=0.5, edgecolor=almost_black, facecolor='green', linewidth=0.15)

    for i, txt in enumerate(ngram_text):
        ax.annotate(txt, (x[i], y[i]))

    x = [candidates_tsne[i, 0] for i in range(1, num_positives+1)]
    y = [candidates_tsne[i, 1] for i in range(1, num_positives+1)]

    ngram_text = [str(ranks[i]) + " (" + all_candidates_tag[i] + ")" for i in range(num_positives)]
    ax.scatter(x, y, label='positive', alpha=0.5, edgecolor=almost_black, facecolor='red', linewidth=0.15)

    for i, txt in enumerate(ngram_text):
        ax.annotate(txt, (x[i], y[i]))

    x = [candidates_tsne[i, 0] for i in range(num_positives+1, len(all_candidates_tag)+1)]
    y = [candidates_tsne[i, 1] for i in range(num_positives+1, len(all_candidates_tag)+1)]

    ngram_text = [str(ranks[i]) + " (" + all_candidates_tag[i] + ")" for i in range(num_positives, len(all_candidates_tag))]
    ax.scatter(x, y, label='negatives', alpha=0.5, edgecolor=almost_black, facecolor='blue', linewidth=0.15)

    for i, txt in enumerate(ngram_text):
        ax.annotate(txt, (x[i], y[i]))

    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)

    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)

    ax.title.set_color(almost_black)

    light_grey = np.array([float(248)/float(255)]*3)
    legend = ax.legend(frameon=True, scatterpoints=1)
    rect = legend.get_frame()
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)

    # Change the legend label colors to almost black, too
    texts = legend.texts
    for t in texts:
        t.set_color(almost_black)

    ax.grid(False)
    fig.savefig(str(file_path), dpi=200)
    plt.close()


def arrau_sample(batches):
    #random_num = random.sample(range(len(batches)), 1)
    #batch = batches[random_num[0]]
    batch = batches[0]
    for b in batches[1:]:
        batch.extend(b)

    anaphor_1, sent_pa_1, \
    positive_candidates_1, negative_candidates_1, \
    positive_candidates_tag_1, negative_candidates_tag_1, \
    sent_anaph_len_1, positive_candidates_len_1, negative_candidates_len_1, \
    sent_pa_tag_1, num_positives_1, num_negatives_1, ctx_all_1, ctx_len_1 = zip(*batch)

    duplicates = []
    for j, anaphs1 in enumerate(sent_pa_1):
        indices = [i for i, anaphs2 in enumerate(sent_pa_1) if anaphs2[:sent_anaph_len_1[i]] == anaphs1[:sent_anaph_len_1[j]]]
        if len(indices) > 1:
            if indices not in duplicates:
                duplicates.append(indices)
    if duplicates:
        duplicates_sum = sum([len(x) for x in duplicates])
        logging.info('number of duplicated anaphoric sentences: %s' % duplicates_sum)
        data_sample = []
        for indx in duplicates[2]:
            data_sample.append(batch[indx])
        return data_sample

    #else:
        #data_sample = arrau_sample(batches)
        #return data_sample
    return None


def eval(argv):
    checkpoint_file = tf.train.latest_checkpoint(argv.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            sent_pa_pl = graph.get_operation_by_name("placeholders/anaph_sent").outputs[0]
            sent_pa_len_pl = graph.get_operation_by_name("placeholders/anaphs_len").outputs[0]
            positive_candidates_pl = graph.get_operation_by_name("placeholders/positive_candidates").outputs[0]
            positive_candidates_len_pl = graph.get_operation_by_name("placeholders/positive_candidates_ln").outputs[0]
            negative_candidates_pl = graph.get_operation_by_name("placeholders/negative_candidates").outputs[0]
            negative_candidates_len_pl = graph.get_operation_by_name("placeholders/negative_candidates_len").outputs[0]
            anaphors_pl = graph.get_operation_by_name("placeholders/anaphors").outputs[0]
            sent_pa_tag_pl = graph.get_operation_by_name("placeholders/anaphs_tag").outputs[0]
            positive_candidates_tag_pl = graph.get_operation_by_name("placeholders/positive_candidates_tag").outputs[0]
            negative_candidates_tag_pl = graph.get_operation_by_name("placeholders/negative_candidates_tag").outputs[0]
            num_positives_pl = graph.get_operation_by_name("placeholders/real_num_positives").outputs[0]
            num_negatives_pl = graph.get_operation_by_name("placeholders/real_num_negatives").outputs[0]
            ctx_pl = graph.get_operation_by_name("placeholders/ctx").outputs[0]
            ctx_len_pl =  graph.get_operation_by_name("placeholders/ctx_len").outputs[0]
            keep_rate_input_pl =  graph.get_operation_by_name("placeholders/keep_rate_input").outputs[0]
            keep_rate_cell_output_pl = graph.get_operation_by_name("placeholders/keep_rate_cell_output").outputs[0]
            keep_ffl1_rate_pl = graph.get_operation_by_name("placeholders/self.keep_ffl1_rate").outputs[0]
            keep_ffl2_rate_pl = graph.get_operation_by_name("placeholders/self.keep_ffl2_rate").outputs[0]

            # Tensors we want to evaluate
            scores_op = graph.get_operation_by_name("siamese/scores/scores").outputs[0]
            anaphs_elu_op = graph.get_operation_by_name("siamese/LSTM/anaphs_elu").outputs[0]
            positives_elu_op = graph.get_operation_by_name("siamese/LSTM/positives_elu").outputs[0]
            negatives_elu_op = graph.get_operation_by_name("siamese/LSTM/negatives_elu").outputs[0]
            joint_positives_elu_op = graph.get_operation_by_name("siamese/scores/joint_positives_elu").outputs[0]
            joint_negatives_elu_op = graph.get_operation_by_name("siamese/scores/joint_negatives_elu").outputs[0]
            output_states_anaphs_op = graph.get_operation_by_name("siamese/LSTM/output_states_anaphs").outputs[0]

            precision_at_n = [0.0]*4

            vocabulary_inv = [""]*len(argv.vocabulary)
            for w in argv.vocabulary:
                vocabulary_inv[argv.vocabulary[w]] = w

            test_sample = arrau_sample(argv.test_batches)

            hm_matrices = []

            for k, batch in enumerate(test_sample):
                anaphor_1, sent_pa_1,\
                positive_candidates_1, negative_candidates_1,\
                positive_candidates_tag_1, negative_candidates_tag_1,\
                sent_anaph_len_1, positive_candidates_len_1, negative_candidates_len_1,\
                sent_pa_tag_1, num_positives_1, num_negatives_1, ctx_all_1, ctx_len_1 = batch


                anaphors = [anaphor_1]
                sent_pa = [sent_pa_1]
                positive_candidates = [positive_candidates_1]
                negative_candidates = [negative_candidates_1]
                positive_candidates_len = [positive_candidates_len_1]
                negative_candidates_len = [negative_candidates_len_1]
                positive_candidates_tag = [positive_candidates_tag_1]
                negative_candidates_tag = [negative_candidates_tag_1]
                sent_anaph_len = [sent_anaph_len_1]
                sent_pa_tag = [sent_pa_tag_1]
                num_positives = [num_positives_1]
                num_negatives = [num_negatives_1]
                ctx_all = [ctx_all_1]
                ctx_len = [ctx_len_1]

                feed_dict = {sent_pa_pl: np.asarray(sent_pa, dtype=np.int32),
                             sent_pa_len_pl: np.asarray(sent_anaph_len, dtype=np.int32),
                             positive_candidates_pl: np.asarray(positive_candidates, dtype=np.int32),
                             positive_candidates_len_pl: np.asarray(positive_candidates_len, dtype=np.int32),
                             negative_candidates_pl: np.asarray(negative_candidates, dtype=np.int32),
                             negative_candidates_len_pl: np.asarray(negative_candidates_len, dtype=np.int32),
                             anaphors_pl: np.asarray(anaphors, dtype=np.int32),
                             positive_candidates_tag_pl: np.asarray(positive_candidates_tag, dtype=np.int32),
                             negative_candidates_tag_pl: np.asarray(negative_candidates_tag, dtype=np.int32),
                             sent_pa_tag_pl: np.asarray(sent_pa_tag, dtype=np.int32),
                             num_positives_pl: np.asarray(num_positives, dtype=np.int32),
                             num_negatives_pl: np.asarray(num_negatives, dtype=np.int32),
                             ctx_pl: np.asarray(ctx_all, dtype=np.int32),
                             ctx_len_pl: np.asarray(ctx_len, dtype=np.int32),
                             keep_rate_input_pl: 1.0,
                             keep_rate_cell_output_pl: 1.0,
                             keep_ffl1_rate_pl: 1.0,
                             keep_ffl2_rate_pl: 1.0}

                test_scores,\
                output_states_anaphs,\
                anaphs_elu,\
                positives_elu,\
                negatives_elu,\
                joint_positives_elu,\
                joint_negatives_elu = sess.run([scores_op,
                                          output_states_anaphs_op,
                                          anaphs_elu_op,
                                          positives_elu_op,
                                          negatives_elu_op,
                                          joint_positives_elu_op,
                                          joint_negatives_elu_op],
                                         feed_dict)
                pn_batch = precision_n(test_scores, num_positives, 4)
                print pn_batch
                precision_at_n = map(add, pn_batch, precision_at_n)

                ##### everything in comments is for the heatmap #####

                '''
                output_states_anaphs_tp = np.asarray(output_states_anaphs).transpose()
                output_states_anaphs_org = output_states_anaphs_tp.reshape(2*argv.hidden_size,len(sent_pa_1))
                output_states_anaphs = output_states_anaphs_org[:, :sent_anaph_len_1]
                '''

                anaphs_elu = anaphs_elu.reshape((1, argv.hidden_size_ffl1))
                anaphs_elu = np.concatenate((anaphs_elu, np.zeros((1, argv.hidden_size_ffl2 - argv.hidden_size_ffl1))), 1)
                anaphs_elu = anaphs_elu.reshape((1, argv.hidden_size_ffl2))
                positives_elu = positives_elu.reshape((num_positives_1, argv.hidden_size_ffl1))
                negatives_elu = negatives_elu.reshape((num_negatives_1, argv.hidden_size_ffl1))

                joint_positives_elu = joint_positives_elu.reshape((num_positives_1, argv.hidden_size_ffl2))
                joint_negatives_elu = joint_negatives_elu.reshape((num_negatives_1, argv.hidden_size_ffl2))

                pos_tags_filename = "../corpora/par_data/up2date_data/penn_treebank_tags.txt"
                pos_tags_lines = codecs.open(pos_tags_filename, "r", encoding="utf-8").readlines()
                pos_tags = [tag.split("\n")[0] for tag in pos_tags_lines]
                pos_ids = range(len(pos_tags))
                pos_vocabulary = dict(zip(pos_tags, pos_ids))
                pos_vocabulary[UNK] = len(pos_tags)

                all_candidates_tag_ids = positive_candidates_tag_1 + negative_candidates_tag_1
                all_candidates_tag = [pos_tags[tid] for tid in all_candidates_tag_ids ]

                #outputs = np.concatenate((anaphs_elu, positives_elu, negatives_elu), 0)
                joint = np.concatenate((anaphs_elu, joint_positives_elu, joint_negatives_elu), 0)
                tsne = TSNE(init='pca', n_iter=3000)
                candidates_tsne = tsne.fit_transform(joint)
                filename = "figs/tsne_" + str(k) + ".png"
                item = test_scores[0]
                ranks = len(item) - rankdata(item, method='ordinal').astype(int)
                plot(candidates_tsne, all_candidates_tag, num_positives_1, filename, ranks)


                '''
                sent_pa_string = []
                for wid in sent_pa_1:
                    sent_pa_string.append(vocabulary_inv[wid])
                sent_pa_string_org = sent_pa_string[:sent_anaph_len_1]
                hm_matrices.append(output_states_anaphs)
                indices = []
                ctx_clean = [c for c in ctx_all_1 if c != 14026]
                for i in range(len(sent_pa_1)):
                    if sent_pa_1[i:i + len(ctx_clean)] == ctx_clean:
                        indices = range(i, i + len(ctx_clean))

                sent_pa_string_clean = []
                for i, w in enumerate(sent_pa_string_org):
                    if i in indices:
                        sent_pa_string_clean.append(w)
                    else:
                        sent_pa_string_clean.append("")

                fig, ax = ppl.subplots(1)
                fig.set_figheight(15)
                fig.set_figwidth(30)
                ppl.pcolormesh(fig, ax, output_states_anaphs)
                ax.set_xticks(np.arange(0.5, len(sent_pa_string_org) + 0.5, 1))
                ax.set_xticklabels(sent_pa_string_org, size=15)
                fig_path = "figs/heatmap_" + str(k) + ".png"
                fig.savefig(fig_path)
                '''

            precision_at_n[:] = [x / (float(len(argv.test_batches))) for x in precision_at_n]
            logging.info('Test precision at 4:')
            logging.info(precision_at_n)

            '''
            difference = np.absolute(hm_matrices[0] - hm_matrices[1])

            fig, ax = ppl.subplots(1)
            fig.set_figheight(15)
            fig.set_figwidth(30)
            ppl.pcolormesh(fig, ax, difference
                           )
            ax.set_xticks(np.arange(0.5, len(sent_pa_string_org) + 0.5, 1))
            ax.set_xticklabels(sent_pa_string_org, size=15)
            fig_path = "figs/heatmap_difference.png"
            fig.savefig(fig_path)
            '''

            return precision_at_n


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
            precision += min(1, len(set(ranks[:num_true[k]]) & set(range(i))))
            print num_true[k]
            print ranks
        precision /= float(len(test_scores))
        precision *= 100
        precisions.append(precision)
    return precisions



