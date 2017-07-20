from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
from collections import defaultdict
from collections import Counter
import numpy as np
import logging
import itertools
import ijson
import json
import codecs
import string
import random
import sys

PAD = "<PAD>"
UNK = "<UNK>"


def word2vec_emb_vocab(vocabulary, dim, emb_type):
    '''
    Collect pre-trained embeddings for words in the vocabulary.
    :param vocabulary:
    :param dim: the embedding dimension
    :param emb_type: type of pre-trained embeddings: w2v, glove
    :return:
    '''
    global UNK
    global PAD

    if emb_type == "w2v":
        logging.info("Loading pre-trained w2v binary file...")
        w2v_model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    else:
        # convert glove vecs into w2v format: https://github.com/manasRK/glove-gensim/blob/master/glove-gensim.py
        glove_file = "embeddings/glove/glove_" + str(dim) + "_w2vformat.txt"
        w2v_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)  # GloVe Model

    w2v_vectors = w2v_model.syn0

    vocab_size = len(vocabulary)
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)

    embeddings[vocabulary[PAD],:] = np.zeros((1, dim))
    embeddings[vocabulary[UNK],:] = np.mean(w2v_vectors, axis=0).reshape((1, dim))

    counter = 0
    for word in vocabulary:
        try:
            embeddings[vocabulary[word], :] = w2v_model[word].reshape((1, dim))
        except KeyError:
            counter += 1
            embeddings[vocabulary[word], :] = embeddings[vocabulary[UNK],:]

    logging.info("number of out-of-vocab words: %s from %s" % (counter, vocab_size))

    del w2v_model
    del w2v_vectors

    assert len(vocabulary) == embeddings.shape[0]

    return embeddings, vocabulary


def get_emb(emb_type, vocabulary, dim):
    if emb_type == "w2v":
        emb, vocab = word2vec_emb_vocab(vocabulary, dim, emb_type)

    if emb_type == "glove":
        emb, vocab = word2vec_emb_vocab(vocabulary, dim, emb_type)
    return emb, vocab


def generate_training_jsons(datasets, candidates_list_size):
    '''
    Prepare data for the LSTM-Siamese mention-rankking model.
    :param datasets: 'csn', 'artifical_v02'
    :param candidate_list_size: list, with values in ['small', 'big_0', 'big_1, 'big_2', 'big_3']
                                small if candidates extracted from sentences that contain antecedents,
                                big_0 from the anaphoric sentence,
                                big_x from the anaphoric sentence and x preceding sentence, x >= 1
    :return: dictionary
    '''
    global PAD
    global UNK

    for dataset in datasets:
        for size in candidates_list_size:
            if dataset.split("_")[0] == "csn":
                filename = "data/csn.json"
            else:
                filename = "data/" + dataset + ".json"

            sentences = [] # all text sequences for constructing vocabulary
            sent_anaph = [] # anaphoric sentence
            anaph = [] # anaphor / shell noun
            ctx_all = [] # context of the anaphor/shell noun
            positive_candidates_all = []
            negative_candidates_all = []
            positive_candidates_tag_all = []
            negative_candidates_tag_all = []

            for item in ijson.items(open(filename, 'r'), "item"):
                anaphoric_sentence = word_tokenize(item['artificial_source_suggestion'].lower().replace("-x-", ""))

                # ignore anaphoric sentences with length (in tokens) < 10
                if len(anaphoric_sentence) < 10:
                    continue

                # shell nouns are trained on individual training data
                if dataset.split("_")[0] == "csn" and item['sbar_head'].lower().split(" ")[-1] != dataset.split("_")[1]:
                    continue

                positive_candidates_dpl = [candidate.lower() for candidate in item['artificial_antecedent']]
                assert positive_candidates_dpl

                try:
                    positive_candidates_tag_dpl = item['antecedent_all_node']
                except KeyError:
                    positive_candidates_tag_dpl = item['artificial_antecedent_node']

                # remove duplicates (same string, same tag)
                temp_positives = list(set(list(zip(positive_candidates_dpl, positive_candidates_tag_dpl))))
                positive_candidates_list, positive_candidates_tag_list = zip(*temp_positives)

                # tokenize: word_tokenize ignores extra whitespaces
                positive_candidates_tokenize = []
                for candidate in positive_candidates_list:
                    positive_candidates_tokenize.append(word_tokenize(candidate.lower()))
                    sentences.append(word_tokenize(candidate.lower()))

                # among positives can not be instances with same string and different tag
                indices = []
                for i, candidate in enumerate(positive_candidates_tokenize):
                    if i not in indices:
                        candidate_indx = [j for j, x in enumerate(positive_candidates_tokenize) if x == candidate]
                        indices.extend(candidate_indx[1:])

                if len(indices) != 0:
                    continue

                positive_candidates_set = positive_candidates_tokenize
                positive_candidates_tag_set = positive_candidates_tag_list
                assert positive_candidates_set

                if size == "small":
                    negative_candidates_dpl = [candidate.lower() for candidate in item['candidates_minus_all_antecedents']]
                    negative_candidates_tag_dpl = item['candidates_nodes_minus_all_antecedents']

                if size != "small":
                    negative_candidates_dpl = []
                    negative_candidates_tag_dpl = []
                    if size.split("_")[0] == "big":
                        for i in range(int(size.split("_")[1])+1):
                            try:
                                negative_candidates_dpl.extend([candidate.lower() for candidate in item['candidates_'+str(i)+'_minus_all_antecedents']])
                                negative_candidates_tag_dpl.extend(item['candidates_nodes_'+str(i)+'_minus_all_antecedents'])
                            except KeyError:
                                pass

                if not negative_candidates_dpl:
                    continue

                # remove duplicates: same string and tag
                temp_negatives = list(set(list(zip(negative_candidates_dpl, negative_candidates_tag_dpl))))
                negative_candidates_list, negative_candidates_tag_list = zip(*temp_negatives)

                # tokenize: word_tokenize
                negative_candidates_tokenize = []
                for candidate in negative_candidates_list:
                    negative_candidates_tokenize.append(word_tokenize(candidate.lower()))
                    sentences.append(word_tokenize(candidate.lower()))

                # remove duplicates: same string and different tag
                # take the first occuring string and the corresponding tag
                indices = []
                for i, candidate in enumerate(negative_candidates_tokenize):
                    if i not in indices:
                        candidate_indx = [j for j, x in enumerate(negative_candidates_tokenize) if x == candidate]
                        indices.extend(candidate_indx[1:])

                negative_candidates_set = [x for j, x in enumerate(negative_candidates_tokenize) if j not in indices]
                negative_candidates_tag_set = [x for j, x in enumerate(negative_candidates_tag_list) if j not in indices]

                if not negative_candidates_set:
                    continue

                if dataset.split("_")[0] == "csn":
                    head_clean = dataset.split("_")[1]

                elif dataset.split("_")[0] != "csn":
                    # check: the head of the anaphor is not an empty string or punctuation
                    anaphoric_sentence = item['artificial_source_suggestion'].replace("-X-", " -X- ")
                    anaph_sent_token = word_tokenize(anaphoric_sentence)
                    mark_ids = [i for i, x in enumerate(anaph_sent_token) if x == "-X-"]
                    assert len(mark_ids) == 2
                    exclude = set(string.punctuation)
                    head_clean = " ".join([x.lower() for x in anaph_sent_token[max(0, mark_ids[0]+1): min(mark_ids[1], len(anaph_sent_token))] if x not in exclude])
                    if not (head_clean and head_clean != "none" and word_tokenize(head_clean)):
                        continue

                anaph.append(word_tokenize(head_clean))
                anaphoric_sentence = item['artificial_source_suggestion'].replace("-X-", " -X- ")
                anaph_sent_token = word_tokenize(anaphoric_sentence)
                mark_ids = [i for i, x in enumerate(anaph_sent_token) if x == "-X-"]

                # only one anaphor should be marked
                if len(mark_ids) > 2:
                    continue

                anaph_sent_token.remove("-X-")
                anaph_sent_token.remove("-X-")

                ctx = [x.lower() for x in anaph_sent_token[max(0, mark_ids[0] - 1): min(mark_ids[1], len(anaph_sent_token))]]
                ctx_all.append(ctx)
                sent_anaph.append(word_tokenize(anaphoric_sentence))
                sentences.append(word_tokenize(anaphoric_sentence))
                negative_candidates_all.append(negative_candidates_set)
                negative_candidates_tag_all.append(negative_candidates_tag_set)
                positive_candidates_all.append(positive_candidates_set)
                positive_candidates_tag_all.append(positive_candidates_tag_set)

            data = zip(anaph, sent_anaph,
                       positive_candidates_all, negative_candidates_all,
                       positive_candidates_tag_all, negative_candidates_tag_all, ctx_all)
            assert data
            dict_train = {'dataset_sentences': sentences,
                          'data': data}
            with open("../corpora/par_dicts/" + dataset + "_" + size + '.json', 'w') as fp:
                json.dump(dict_train, fp)


def generate_evaluation_jsons(dataset, candidates_list_size):
    '''
    Prepare data for the LSTM-Siamese mention-rankking model.
    :param datasets: 'asn', 'arrau', 'arrau_nominal', 'arrau_pronominal'
    :param candidate_list_size: list, with values in ['small', 'big_0', 'big_1, 'big_2', 'big_3']
                                small if candidates extracted from sentences that contain antecedents,
                                big_0 from the anaphoric sentence,
                                big_x from the anaphoric sentence and x preceding sentence, x >= 1
    :return: dictionary
    '''
    global PAD
    global UNK

    for size in candidates_list_size:
        logging.info("parsing " + dataset + ", " + size + " json...")
        if dataset.split("_")[0] == "asn":
            filename = 'data/asn.json'
        else:
            filename = "data/arrau.json"

        sent_anaph = []
        positive_candidates_all = []
        negative_candidates_all = []
        anaph = []
        count_irr = 0
        positive_candidates_tag_all = []
        negative_candidates_tag_all = []
        ctx_all = []
        distances_count = 0
        outside_count = 0
        distances = []

        for item in ijson.items(open(filename, 'r'), "item"):
            if dataset.split("_")[0] == "asn" and item['anaphor'].lower().split(" ")[-1] != dataset.split("_")[1]:
                continue

            if dataset.split("_")[0] == "asn":
                anaphor = dataset.split("_")[1]
                anaph.append([anaphor])
                anaph_sent_token = word_tokenize(item['anaphor_sentence'])
                anaph_indx = anaph_sent_token.index(dataset.split("_")[1])
                ctx = [x.lower() for x in anaph_sent_token[anaph_indx-1:anaph_indx+2]]
                ctx_all.append(ctx)

            elif dataset.split("_")[0] == "arrau":
                try:
                    if dataset.split("_")[1] != item["anaphor_function"]:
                        continue
                except IndexError:
                    pass

                distance = item['antecedent_distances']
                distances.extend(distance)

                antec_mask = []
                if size != "small":
                    for d in distance:
                        if d <= int(size.split("_")[1]):
                            antec_mask.append(True)
                        else:
                            antec_mask.append(False)
                    if True not in antec_mask:
                        outside_count +=1
                        continue
                if size == "small":
                    antec_mask = [True]*len(distance)

                anaph_sent_token = word_tokenize(item['anaphor_sentence'].replace("-X-", " -X- "))
                mark_ids = [i for i, x in enumerate(anaph_sent_token) if x == "-X-"]
                assert len(mark_ids) == 2
                anaph_sent_token.remove("-X-")
                anaph_sent_token.remove("-X-")

                ctx = [x.lower() for x in
                       anaph_sent_token[max(0, mark_ids[0] - 1): min(mark_ids[1], len(anaph_sent_token))]]
                ctx_all.append(ctx)
                anaph.append(word_tokenize(head_clean))

            sent_anaph.append(word_tokenize(item['anaphor_sentence'].lower().replace("-x-","")))

            try:
                assert item['antecedent_all']
                positive_candidates_dpl = [candidate.lower() for candidate, mask in zip(item['antecedent_all'], antec_mask) if mask]

            except KeyError:
                assert item['antecedent']
                positive_candidates_dpl = [candidate.lower() for candidate, mask in zip(item['antecedent'], antec_mask) if mask]

            if not positive_candidates_dpl:
                continue

            positive_candidates_tag_dpl = item['antecedent_all_node']

            # check: duplicates (same string, same tag)
            temp_positives = list(set(list(zip(positive_candidates_dpl, positive_candidates_tag_dpl))))
            positive_candidates_list, positive_candidates_tag_list = zip(*temp_positives)

            # tokenize: word_tokenize ignores extra whitespaces
            positive_candidates_tokenize = []
            for candidate in positive_candidates_list:
                positive_candidates_tokenize.append(word_tokenize(candidate.lower()))

            # remove duplicates: same string, different tag
            indices = []
            for i, candidate in enumerate(positive_candidates_tokenize):
                if i not in indices:
                    candidate_indx = [j for j, x in enumerate(positive_candidates_tokenize) if x == candidate]
                    indices.extend(candidate_indx[1:])

            assert len(indices) == 0

            positive_candidates_set = [x for j, x in enumerate(positive_candidates_tokenize) if j not in indices]
            positive_candidates_tag_set = [x for j, x in enumerate(positive_candidates_tag_list) if j not in indices]

            if size == 'small':
                negative_candidates_dpl = [candidate.lower() for candidate in item['candidates_minus_all_antecedents']]
                negative_candidates_tag_dpl = item['candidates_nodes_minus_all_antecedents']

            if size != 'small':
                negative_candidates_dpl = []
                negative_candidates_tag_dpl = []
                if size.split("_")[0] == "big":
                    for i in range(int(size.split("_")[1])+1):
                        try:
                            negative_candidates_dpl.extend([candidate.lower() for candidate in item['candidates_' + str(i) + '_minus_all_antecedents']])
                            negative_candidates_tag_dpl.extend(
                                item['candidates_nodes_' + str(i) + '_minus_all_antecedents'])
                        except KeyError:
                            pass

            if not negative_candidates_dpl:
                continue

            # remove duplicates
            temp_negatives = list(set(list(zip(negative_candidates_dpl, negative_candidates_tag_dpl))))
            negative_candidates_list, negative_candidates_tag_list = zip(*temp_negatives)

            # tokenize: word_tokenize
            negative_candidates_tokenize = []
            for candidate in negative_candidates_list:
                negative_candidates_tokenize.append(word_tokenize(candidate.lower()))

            # remove duplicates
            indices = []
            for i, candidate in enumerate(negative_candidates_tokenize):
                if i not in indices:
                    candidate_indx = [j for j, x in enumerate(negative_candidates_tokenize) if x == candidate]
                    indices.extend(candidate_indx[1:])

            negative_candidates_set = [x for j, x in enumerate(negative_candidates_tokenize) if j not in indices]
            negative_candidates_tag_set = [x for j, x in enumerate(negative_candidates_tag_list) if j not in indices]

            if len(positive_candidates_set) > len(negative_candidates_set):
                count_irr += 1

            # add to positives candidates that differ in one word or one word and puncuation
            if dataset.split("_")[0] == "arrau":
                positive_candidates_temp = positive_candidates_set
                positive_candidates_tag_temp = positive_candidates_tag_set
                indices = []

                for i, (negative, negative_tag) in enumerate(
                        zip(negative_candidates_set, negative_candidates_tag_set)):
                    for positive, positive_tag in zip(positive_candidates_temp, positive_candidates_tag_temp):
                        sym_difference = list(set(positive) ^ set(negative))
                        punctuation = list(string.punctuation)
                        intersection_strip = [s for s in sym_difference if s not in punctuation]
                        if len(intersection_strip) <= 1:
                            indices.append(i)
                            positive_candidates_set.append(negative)
                            positive_candidates_tag_set.append(negative_tag)

                negative_candidates_set_final = [x for i, x in enumerate(negative_candidates_set) if i not in indices]
                negative_candidates_tag_set_final = [x for i, x in enumerate(negative_candidates_tag_set) if i not in indices]

            positives_string = [" ".join(pos) for pos in positive_candidates_set]
            negatives_string = [" ".join(neg) for neg in negative_candidates_set_final]

            assert len(set(positives_string) & set(negatives_string)) == 0

            positive_candidates_all.append(positive_candidates_set)
            positive_candidates_tag_all.append(positive_candidates_tag_set)
            negative_candidates_all.append(negative_candidates_set_final)
            negative_candidates_tag_all.append(negative_candidates_tag_set_final)

        dict_eval = {'data': zip(anaph, sent_anaph,
                                 positive_candidates_all, negative_candidates_all,
                                 positive_candidates_tag_all, negative_candidates_tag_all,
                                 ctx_all)}

        with open("par_dicts/" + dataset + "_" + size + '.json', 'w') as fp:
            json.dump(dict_eval, fp)
    data_file.close()


def get_data_from_json(vocabulary, pos_vocabulary, data):
    '''
    Final preparation of the data for the LSTM-Siamese mention ranking model: words -> vocab ids
    '''
    global UNK
    global PAD
    anaph_v1, sent_anaph_v1,\
    positive_candidates_v1, negative_candidates_v1,\
    positive_candidates_tag_v1, negative_candidates_tag_v1, ctx_all = zip(*data)

    out_of_vocab_words = defaultdict(int)
    out_of_vocab_pos = defaultdict(int)
    anaph = []
    for sent in anaph_v1:
        try:
            anaph.append(vocabulary[sent[-1]])
        except KeyError:
            anaph.append(vocabulary[UNK])
            out_of_vocab_words[sent[-1]] += 1

    ctx = []
    for c in ctx_all:
        ctx_tmp = []
        for w in c:
            try:
                ctx_tmp.append(vocabulary[w])
            except KeyError:
                ctx_tmp.append(vocabulary[UNK])
                out_of_vocab_words[w] += 1
        ctx.append(ctx_tmp)

    sent_anaph = []
    for sent in sent_anaph_v1:
        sent_ids = []
        for w in sent:
            try:
                sent_ids.append(vocabulary[w])
            except KeyError:
                sent_ids.append(vocabulary[UNK])
                out_of_vocab_words[w] += 1
        sent_anaph.append(sent_ids)

    positive_candidates = []
    for candidates_item in positive_candidates_v1:
        candidates_temp = []
        for sent in candidates_item:
            sent_ids = []
            for w in sent:
                try:
                    sent_ids.append(vocabulary[w])
                except KeyError:
                    out_of_vocab_words[w] += 1
                    sent_ids.append(vocabulary[UNK])
            candidates_temp.append(sent_ids)
        positive_candidates.append(candidates_temp)

    negative_candidates = []
    for candidates_item in negative_candidates_v1:
        candidates_temp = []
        for sent in candidates_item:
            sent_ids = []
            for w in sent:
                try:
                    sent_ids.append(vocabulary[w])
                except KeyError:
                    out_of_vocab_words[w] += 1
                    sent_ids.append(vocabulary[UNK])
            candidates_temp.append(sent_ids)
        negative_candidates.append(candidates_temp)

    positive_candidates_tag = []
    for candidates_tags in positive_candidates_tag_v1:
        candidates_temp = []
        for tag in candidates_tags:
            try:
                candidates_temp.append(pos_vocabulary[tag])
            except KeyError:
                out_of_vocab_pos[tag] += 1
                candidates_temp.append(pos_vocabulary[UNK])
        positive_candidates_tag.append(candidates_temp)


    negative_candidates_tag = []
    for candidates_tags in negative_candidates_tag_v1:
        candidates_temp = []
        for tag in candidates_tags:
            try:
                candidates_temp.append(pos_vocabulary[tag])
            except KeyError:
                out_of_vocab_pos[tag] += 1
                candidates_temp.append(pos_vocabulary[UNK])
        negative_candidates_tag.append(candidates_temp)

    data = zip(anaph, sent_anaph, positive_candidates, negative_candidates, positive_candidates_tag, negative_candidates_tag, ctx)

    return data, vocabulary


def get_batches_eval(data, batch_size, vocabulary, pos_vocabulary):
    '''
    To implement the model efficiently in TensorFlow, batches are constructed in such a way that every sentence instance
    in the batch has the same number of antecedents and the same number of negative candidates.
    Note that by this we do not mean that the ratio of positive and negative examples is 1:1.
    '''
    data.sort(key=lambda s: len(s[3]))
    group_by_neg = itertools.groupby(data, lambda x: len(x[3]))
    batches = []
    for key1, group1 in group_by_neg:
        g1 = list(group1)
        g1.sort(key=lambda s: len(s[2]))
        group_by_pos = itertools.groupby(g1, lambda x: len(x[2]))
        for key2, group2 in group_by_pos:
            g2 = list(group2)
            size_g2 = len(g2)
            if size_g2 % float(batch_size) == 0:
                num_batches = int(size_g2 / batch_size)
            else:
                num_batches = int(size_g2 / batch_size) + 1

            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(g2))

                batch = pad_batch(g2[start_index:end_index], vocabulary, pos_vocabulary)
                batches.append(batch)
    return batches


def get_batches(data, batch_size, vocabulary, pos_vocabulary):
    '''
    Get batches without any restrictions on number of antecedents and negative candidates.
    '''
    random.seed(24)
    random.shuffle(data)

    data_size = len(data)
    if data_size % float(batch_size) == 0:
        num_batches = int(data_size / float(batch_size))
    else:
        num_batches = int(data_size / float(batch_size)) + 1

    batches = []
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        batch = pad_batch(data[start_index:end_index], vocabulary, pos_vocabulary)
        batches.append(batch)

    logging.info('Data size: %s' % len(data))
    logging.info('Number of batches: %s' % len(batches))

    return batches


def pad_batch(batch, vocab, pos_vocabulary):
    '''
    Pad anaphoric sentences, antecedents, negative candidates to the corresponding maximum length of the batch.
    '''
    global PAD
    global UNK
    pad_id = vocab[PAD]

    anaph, sent_anaph, positive_candidates_org, negative_candidates_org, positive_candidates_tag_org, negative_candidates_tag_org, ctx_all = zip(*batch)
    sent_anaph = list(sent_anaph)
    positive_candidates_org = list(positive_candidates_org)
    negative_candidates_org = list(negative_candidates_org)

    max_length = 0

    max_num_positives = max(len(positives) for positives in positive_candidates_org)
    max_num_negatives = max(len(negatives) for negatives in negative_candidates_org)

    max_ctx_len = max(len(ctx) for ctx in ctx_all)
    ctx_pad = []
    ctx_len = []
    for ctx in ctx_all:
        diff = max_ctx_len - len(ctx)
        ctx_len.append(len(ctx))
        for _ in range(diff):
            ctx.append(vocab[PAD])
        ctx_pad.append(ctx)

    num_positives = []
    num_negatives = []
    for i in range(len(positive_candidates_org)):
        diff = max_num_positives - len(positive_candidates_org[i])
        num_positives.append(len(positive_candidates_org[i]))
        temp = [[vocab[UNK]] for _ in range(diff)]
        temp_tag = [pos_vocabulary[UNK] for _ in range(diff)]
        positive_candidates_org[i].extend(temp)
        positive_candidates_tag_org[i].extend(temp_tag)

    for i in range(len(negative_candidates_org)):
        diff = max_num_negatives - len(negative_candidates_org[i])
        num_negatives.append(len(negative_candidates_org[i]))
        temp = [[vocab[UNK]] for _ in range(diff)]
        temp_tag = [pos_vocabulary[UNK] for _ in range(diff)]
        negative_candidates_org[i].extend(temp)
        negative_candidates_tag_org[i].extend(temp_tag)

    for sent in sent_anaph:
        max_length = max(max_length, len(sent))

    for cand_inst in positive_candidates_org:
        for c in cand_inst:
            max_length = max(max_length, len(c))

    for cand_inst in negative_candidates_org:
        for c in cand_inst:
            max_length = max(max_length, len(c))

    sent_anaph_pad_all = []
    sent_anaph_len = []

    positive_candidates_pad_all = []
    positive_candidates_len = [[] for _ in range(len(sent_anaph))]

    negative_candidates_pad_all = []
    negative_candidates_len = [[] for _ in range(len(sent_anaph))]

    for i in range(len(sent_anaph)):
        sent_anaph_inst = sent_anaph[i]
        positive_candidates_inst = positive_candidates_org[i]
        negative_candidates_inst = negative_candidates_org[i]

        sent_anaph_len.append(len(sent_anaph_inst))

        positive_candidates_inst_pad = []
        for candidate in positive_candidates_inst:
            positive_candidates_len[i].append(len(candidate))
            diff = max_length - len(candidate)
            assert diff >= 0
            temp = [pad_id]*diff
            candidate_pad = candidate + temp
            positive_candidates_inst_pad.append(candidate_pad)

        negative_candidates_inst_pad = []
        for candidate in negative_candidates_inst:
            negative_candidates_len[i].append(len(candidate))
            diff = max_length - len(candidate)
            assert diff >= 0
            temp = [pad_id]*diff
            candidate_pad = candidate + temp
            negative_candidates_inst_pad.append(candidate_pad)

        diff = max_length - len(sent_anaph_inst)
        temp = [pad_id]*diff
        sent_anaph_pad = sent_anaph_inst + temp

        sent_anaph_pad_all.append(sent_anaph_pad)
        positive_candidates_pad_all.append(positive_candidates_inst_pad)
        negative_candidates_pad_all.append(negative_candidates_inst_pad)

    sent_anaph_tag = [[pos_vocabulary["S"]]*max_length for _ in range(len(batch))]

    batch_pad = zip(anaph, sent_anaph_pad_all,
                    positive_candidates_pad_all, negative_candidates_pad_all,
                    positive_candidates_tag_org, negative_candidates_tag_org,
                    sent_anaph_len, positive_candidates_len, negative_candidates_len,
                    sent_anaph_tag, num_positives, num_negatives, ctx_pad, ctx_len)
    return batch_pad


def prune_negatives_by_length(data, threshold):
    '''
    Remove negative candidates w.r.t. the length.
    '''
    anaph, sent_anaph, positive_candidates, negative_candidates, positive_candidates_tag, negative_candidates_tag, ctx_all = zip(*data)

    count = 0
    negative_candidates_new = []
    negative_candidates_tag_new = []
    indices = []
    for i, (item, item_tag) in enumerate(zip(negative_candidates, negative_candidates_tag)):
        item_new = []
        item_tag_new = []
        for c, t in zip(item, item_tag):
            if len(c) >= threshold:
                item_new.append(c)
                item_tag_new.append(t)
            else:
                count += 1

        if not item_new:
            indices.append(i)
        else:
            negative_candidates_new.append(item_new)
            negative_candidates_tag_new.append(item_tag_new)
    anaph_new = [a for i, a in enumerate(anaph) if i not in indices]
    sent_anaph_new = [s for i, s in enumerate(sent_anaph) if i not in indices]
    positive_candidates_new = [item for i, item in enumerate(positive_candidates) if i not in indices]
    positive_candidates_tag_new = [item for i, item in enumerate(positive_candidates_tag) if i not in indices]
    ctx_all_new = [ctx for i, ctx in enumerate(ctx_all) if i not in indices]
    prune_count = count / float(len(negative_candidates))

    logging.info('on average %s false candidates per instances were removed' % prune_count)

    data_new = zip(anaph_new, sent_anaph_new, positive_candidates_new, negative_candidates_new,
                   positive_candidates_tag_new, negative_candidates_tag_new, ctx_all_new)

    return data_new


def prune_negatives_by_tag(data):
    '''
    Remove negative candidates that are not sentence worthy.
    '''
    global UNK
    allowed_tags = ["S", "VP", "ROOT", "SBAR", "None", "SBARQ", ""]

    anaph, sent_anaph, positive_candidates, negative_candidates, positive_candidates_tag, negative_candidates_tag, ctx_all = zip(*data)

    count = 0
    negative_candidates_new = []
    negative_candidates_tag_new = []
    indices = []
    for i, (item, item_tag) in enumerate(zip(negative_candidates, negative_candidates_tag)):
        item_new = []
        item_tag_new = []
        for c, t in zip(item, item_tag):
            if t in allowed_tags:
                item_new.append(c)
                item_tag_new.append(t)
            else:
                count += 1
        if not item_new:
            indices.append(i)
        if item_new:
            negative_candidates_new.append(item_new)
            negative_candidates_tag_new.append(item_tag_new)

    anaph_new = [a for i, a in enumerate(anaph) if i not in indices]
    sent_anaph_new = [s for i, s in enumerate(sent_anaph) if i not in indices]
    positive_candidates_new = [item for i, item in enumerate(positive_candidates) if i not in indices]
    positive_candidates_tag_new = [item for i, item in enumerate(positive_candidates_tag) if i not in indices]
    ctx_all_new = [ctx for i, ctx in enumerate(ctx_all) if i not in indices]
    prune_count = count / float(len(negative_candidates))

    logging.info('on average %s false candidates per instances were removed' % prune_count)

    data_new = zip(anaph_new, sent_anaph_new, positive_candidates_new, negative_candidates_new,
                   positive_candidates_tag_new, negative_candidates_tag_new, ctx_all_new)

    return data_new


def get_eval_setup(eval_setup_id, word_freq, train_corpus, candidates_num,
                   pruning_by_len, pruning_by_tag, batch_size, emb_type, emb_size, perc=None):
    '''
    :param eval_setup_id:
    eval_setup_id: train, dev, test
    1: train = artificial small, dev = ASN, test = ARRAU
    5: train=csn shell noun (e.g. csn_fact), dev=ARRAU, test=asn one shell noun (e.g. asn_fact)
    :return: returns train/dev/test batches for a certain train/dev/test evaluation setup
    '''

    if eval_setup_id == 1:
        # train: artificial
        # dev: ASN
        # test: ARRAU
        logging.info('loading training data json file...')

        json_file = '../corpora/par_dicts/' + train_corpus + '_' + candidates_num + '.json'
        with open(json_file) as data_file:
            artificial = json.load(data_file)

        # shuffles data
        random.seed(24)
        artificial_data = random.sample(artificial['data'], len(data))
        # without shuffling of the data => uncomment the next line
        #artificial_data = artificial['data']
        sentences = artificial['dataset_sentences']

        # build vocabulary
        logging.info('building vocabularies...')
        word_counts = dict(Counter(itertools.chain(*sentences)).most_common())
        word_counts_prune = {k: v for k, v in word_counts.iteritems() if v >= word_freq}
        word_counts_list = zip(word_counts_prune.keys(), word_counts_prune.values())

        vocabulary_inv = [x[0] for x in word_counts_list]
        vocabulary_inv.append(PAD)
        vocabulary_inv.append(UNK)
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        # get puncuation vocabulary ids
        punctuation = list(string.punctuation)
        punctuation_ids = []
        for pid in punctuation:
            try:
                punctuation_ids.append(vocabulary[pid])
            except KeyError:
                pass

        # build TAG vocabulary
        pos_tags_filename = "../corpora/par_data/up2date_data/penn_treebank_tags.txt"
        pos_tags_lines = codecs.open(pos_tags_filename, "r", encoding="utf-8").readlines()
        pos_tags = [tag.split("\n")[0] for tag in pos_tags_lines]
        pos_ids = range(len(pos_tags))
        pos_vocabulary = dict(zip(pos_tags, pos_ids))
        pos_vocabulary[UNK] = len(pos_tags)

        # get embeddings
        logging.info('getting embeddings...')
        embeddings, vocabulary = get_emb(emb_type, vocabulary, emb_size)

        # prune negatives by length or/and by tag
        if pruning_by_len > 1:
            logging.info('pruning negatives by length...')
            artificial_data = prune_negatives_by_length(artificial_data, pruning_by_len)

        if pruning_by_tag == "True":
            logging.info('pruning negatives by tag...')
            artificial_data = prune_negatives_by_tag(artificial_data)

        random.seed(24)
        random.shuffle(artificial_data)

        # get train batches
        artificial_data_nn, vocabulary = get_data_from_json(vocabulary, pos_vocabulary, artificial_data)

        logging.info("data size: %s" % len(artificial_data_nn))

        #train_batches = get_batches(artificial_data_nn, batch_size, vocabulary, pos_vocabulary)
        train_batches = get_batches_eval(artificial_data_nn, len(artificial_data_nn), vocabulary, pos_vocabulary)

        #random.shuffle(train_batches)
        # get dev batches
        asn_all_sn = []
        for sn in ["fact", "reason", "issue", "decision", "question", "possibility"]:
            json_data = "../corpora/par_dicts/asn_" + sn + "_" + candidates_num + '.json'
            with open(json_data) as data_file:
                asn = json.load(data_file)
            asn_data = asn['data']

            if pruning_by_len > 1:
                logging.info('pruning negatives by length...')
                asn_data = prune_negatives_by_length(asn_data, pruning_by_len)
            '''
            if pruning_by_tag == "True":
                logging.info('pruning negatives by tag...')
                asn_data = prune_negatives_by_tag(asn_data)
            '''
            asn_sn, _ = get_data_from_json(vocabulary, pos_vocabulary, asn_data)
            asn_sn = list(asn_sn)
            asn_all_sn.extend(asn_sn)

        dev_batches = get_batches_eval(asn_all_sn, len(asn_all_sn), vocabulary, pos_vocabulary)

        # get test batches
        json_data = "par_dicts/arrau_" + candidates_num + '.json'
        with open(json_data) as data_file:
            arrau = json.load(data_file)
        arrau_data = arrau['data']

        if pruning_by_len > 1:
            logging.info('pruning negatives by length...')
            arrau_data = prune_negatives_by_length(arrau_data, pruning_by_len)

        '''
        if pruning_by_tag == "True":
            logging.info('pruning negatives by tag...')
            arrau_data = prune_negatives_by_tag(arrau_data)
        '''

        test_data, vocabulary = get_data_from_json(vocabulary, pos_vocabulary, arrau_data)
        test_batches = get_batches_eval(test_data, len(test_data), vocabulary, pos_vocabulary)

        json_data = "par_dicts/arrau_nominal_" + candidates_num + '.json'
        with open(json_data) as data_file:
            arrau_nominal = json.load(data_file)
        arrau_nominal_data = arrau_nominal['data']

        if pruning_by_len > 1:
            logging.info('pruning negatives by length...')
            arrau_nominal_data = prune_negatives_by_length(arrau_nominal_data, pruning_by_len)
        '''
        if pruning_by_tag == "True":
            logging.info('pruning negatives by tag...')
            arrau_nominal_data = prune_negatives_by_tag(arrau_nominal_data)
        '''

        nominal_test_data, vocabulary = get_data_from_json(vocabulary, pos_vocabulary, arrau_nominal_data)
        nominal_test_batches = get_batches_eval(nominal_test_data, len(nominal_test_data), vocabulary, pos_vocabulary)

        json_data = "par_dicts/arrau_pronominal_" + candidates_num + '.json'
        with open(json_data) as data_file:
            arrau_pronominal = json.load(data_file)
        pronominal_arrau_data = arrau_pronominal['data']

        if pruning_by_len > 1:
            logging.info('pruning negatives by length...')
            pronominal_arrau_data = prune_negatives_by_length(pronominal_arrau_data, pruning_by_len)
        '''
    
        if pruning_by_tag == "True":
            logging.info('pruning negatives by tag...')
            pronominal_arrau_data = prune_negatives_by_tag(pronominal_arrau_data)
        '''

        pronominal_test_data, vocabulary = get_data_from_json(vocabulary, pos_vocabulary, pronominal_arrau_data)
        pronominal_test_batches = get_batches_eval(pronominal_test_data, len(pronominal_test_data), vocabulary, pos_vocabulary)

        return train_batches, dev_batches, test_batches, nominal_test_batches, pronominal_test_batches, embeddings, vocabulary, pos_vocabulary

    if eval_setup_id == 2:
        json_file = '../corpora/par_dicts/' + train_corpus + "_" + candidates_num + '.json'
        with open(json_file) as data_file:
            csn = json.load(data_file)
        csn_data = csn['data']
        sentences = csn['dataset_sentences']

        # build vocabulary
        logging.info('building vocabularies...')
        word_counts = dict(Counter(itertools.chain(*sentences)).most_common())
        word_counts_prune = {k: v for k, v in word_counts.iteritems() if v >= word_freq}
        word_counts_list = zip(word_counts_prune.keys(), word_counts_prune.values())

        vocabulary_inv = [x[0] for x in word_counts_list]
        vocabulary_inv.append(PAD)
        vocabulary_inv.append(UNK)
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        # get puncuation vocabulary ids
        punctuation = list(string.punctuation)
        punctuation_ids = []
        for pid in punctuation:
            try:
                punctuation_ids.append(vocabulary[pid])
            except KeyError:
                pass

        # build TAG vocabulary
        pos_tags_filename = "../corpora/par_data/up2date_data/penn_treebank_tags.txt"
        pos_tags_lines = codecs.open(pos_tags_filename, "r", encoding="utf-8").readlines()
        pos_tags = [tag.split("\n")[0] for tag in pos_tags_lines]
        pos_ids = range(len(pos_tags))
        pos_vocabulary = dict(zip(pos_tags, pos_ids))
        pos_vocabulary[UNK] = len(pos_tags)

        # get embeddings
        logging.info('getting embeddings...')
        embeddings, vocabulary = get_emb(emb_type, vocabulary, emb_size)

        # prune negatives by length or/and by tag
        if pruning_by_len > 1:
            logging.info('pruning negatives by length...')
            csn_data = prune_negatives_by_length(csn_data, pruning_by_len)

        if pruning_by_tag == "True":
            logging.info('pruning negatives by tag...')
            csn_data = prune_negatives_by_tag(csn_data)

        #random.seed(24)
        #random.shuffle(artificial_data)

        # get train batches
        csn_data_nn, vocabulary = get_data_from_json(vocabulary, pos_vocabulary, csn_data)

        logging.info("data size: %s" % len(csn_data_nn))

        #train_batches = get_batches_eval(csn_data_nn, batch_size, vocabulary, pos_vocabulary)
        train_batches = get_batches(csn_data_nn, batch_size, vocabulary, pos_vocabulary)

        #random.shuffle(train_batches)
        # get dev batches
        json_data = "../corpora/par_dicts/arrau_" + candidates_num + '.json'
        with open(json_data) as data_file:
            arrau = json.load(data_file)
        arrau_data = arrau['data']
        dev_data, _ = get_data_from_json(vocabulary, pos_vocabulary, arrau_data)
        dev_batches = get_batches_eval(dev_data, len(dev_data), vocabulary, pos_vocabulary)

        # get test batches
        json_data = "../corpora/par_dicts/asn_" + train_corpus.split("_")[1] + "_" + candidates_num + '.json'
        with open(json_data) as data_file:
            asn = json.load(data_file)
        asn_data = asn['data']
        asn_nn, _ = get_data_from_json(vocabulary, pos_vocabulary, asn_data)
        test_batches = get_batches_eval(asn_nn, len(asn_nn), vocabulary, pos_vocabulary)

        return train_batches, dev_batches, test_batches, embeddings, vocabulary, pos_vocabulary

if __name__ == '__main__':
    train_datasets = ['csn_fact', 'csn_reason', 'csn_possibility', 'csn_question', 'csn_issue', 'csn_decision',
                      'artificial_v02']
    test_datasets = ['asn_fact', 'asn_reason', 'asn_possibility', 'asn_question', 'asn_issue', 'asn_decision',
                     'arrau', 'arrau_nominal', 'arrau_pronominal']

    # for experiments in the paper only 'small' was used
    candidates_list_size = ['small', 'big_0', 'big_1', 'big_2', 'big_3']

    generate_training_jsons(train_datasets, candidates_list_size)
    generate_test_jsons(test_datasets, candidates_list_sizes)

