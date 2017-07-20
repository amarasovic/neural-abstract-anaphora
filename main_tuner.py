#import optunity
#from optunity.solvers import RandomSearch
from hyperopt import hp, fmin, tpe
from math import log
import csv
from time import time
from data_utils import get_eval_setup
from train_tuner import train
import argparse
import logging
import codecs

run_counter = 0


def run_wrapper(params):
    global run_counter
    global output_file

    run_counter += 1
    logging.info("run:  %s" % run_counter)
    s = time()
    dev_et_1 = run_test(params)

    "elapsed: {}s \n".format(int(round(time() - s)))

    writer.writerow([dev_et_1] + list(params))
    output_file.flush()
    return dev_et_1


def run_test(params):
    global argv

    hidden_size, hidden_size_ffl1, hidden_size_ffl2, pos_emb_size, grad_clip, word_freq, reg_coef,\
    keep_rate_cell_output, keep_rate_input, keep_ffl1_rate, keep_ffl2_rate = params

    parser.add_argument('--hidden_size', type=int, default=int(hidden_size), help='the dimension of the LSTM hidden state')
    parser.add_argument('--hidden_size_ffl1', type=int, default=int(hidden_size_ffl1), help='the dimension of the first FF layer')
    parser.add_argument('--hidden_size_ffl2', type=int, default=int(hidden_size_ffl2), help='the dimension of the second FF layer')
    parser.add_argument("--grad_clip", type=float, default=grad_clip, help="clip gradients to this value")
    parser.add_argument('--reg_coef', type=float, default=reg_coef, help='L2 Reg rate')
    parser.add_argument('--word_freq', type=int, default=word_freq, help='word frequency')
    parser.add_argument('--pos_emb_size', type=int, default=int(pos_emb_size), help='the dimension of POS embeddings')
    parser.add_argument('--keep_rate_cell_output', type=float, default=keep_rate_cell_output, help='keep_rate_cell_output')
    parser.add_argument('--keep_rate_input', type=float, default=keep_rate_input, help='keep_rate_input')
    parser.add_argument('--keep_ffl1_rate', type=float, default=keep_ffl1_rate, help='keep_ffl1_rate')
    parser.add_argument('--keep_ffl2_rate', type=float, default=keep_ffl2_rate, help='keep_ffl2_rate')
    trail_id = int(argv.trail_id) + 1
    parser.add_argument("--trail_id", type=str, default=str(trail_id), help="model_id")
    argv = parser.parse_args()

    # Data Preparation
    # ==================================================
    train_batches, \
    dev_batches, \
    test_batches, \
    nominal_test_batches, \
    pronominal_test_batches, \
    embeddings, \
    vocabulary, \
    pos_vocabulary  = get_eval_setup(argv.eval_setup_id,
                                     argv.word_freq,
                                     argv.train_corpus,
                                     argv.candidates_num,
                                     argv.prune_by_len,
                                     argv.prune_by_tag,
                                     argv.batch_size,
                                     argv.emb_type,
                                     argv.emb_size)

    # Training
    # ==================================================
    logging.info("start training...")
    parser.add_argument('--pos_vocabulary', default=pos_vocabulary, help="pos vocabulary")
    parser.add_argument('--embeddings', default=embeddings, help="embeddings matrix")
    parser.add_argument('--train_batches', default=train_batches, help='train batches')
    parser.add_argument('--dev_batches', default=dev_batches, help='dev batches')
    parser.add_argument('--test_batches', default=test_batches, help='test batches')
    parser.add_argument('--nominal_test_batches', default=nominal_test_batches, help='nominal test batches')
    parser.add_argument('--pronominal_test_batches', default=pronominal_test_batches, help='pronominal test batches')
    argv = parser.parse_args()
    dev_et_1 = train(argv)

    return dev_et_1


if __name__ == '__main__':
    # Parameters
    # ==================================================
    parser = argparse.ArgumentParser(description='propositional anaphora resolution', add_help=False,
                                     conflict_handler='resolve')

    parser.add_argument('-mode', default='train', help='train/test')

    """ input """
    parser.add_argument("--eval_setup_id", type=int, default=1, help="eval setup id")
    '''
    eval_setup_id: train, dev, test
    1: artificial small (15K), ASN, ARRAU
    2: ASN, artificial, ARRAU
    3: artificial big (800K), ASN, ARRAU
    4: aritifical mid + ASN, artificial rest, ARRAU
    5: csn shell noun (e.g. csn_fact), ARRAU, asn one shell noun (e.g. asn_fact)
    6: csn, all shell nouns, ARRAU, asn one shell noun
    '''
    parser.add_argument('--candidates_num', default="small", help="small = the candidates extracted only from"
                                                                  "        the sent w/ the true antec,"
                                                                  "big_X = small + candidates extracted from"
                                                                  "                 X preceding sentences")

    parser.add_argument('--train_corpus', default=None, help='corpus for training')
    parser.add_argument("--prune_by_len", type=int, default=1, help="threshold for minimum neg. candidate's length")
    parser.add_argument("--prune_by_tag", default="False", help="prune neg. candidates by tag")

    parser.add_argument('--tag_feature', type=str, default="True", help="use tag feature")
    parser.add_argument('--anaphor_feature', type=str, default="True", help="use anaphor feature")
    parser.add_argument('--ctx_feature', type=str, default="True", help="use ctx anaphor feature")
    parser.add_argument("--pretrained_emb", type=str, default="True", help="use pretrained word embeddings")

    parser.add_argument('--emb_type', default="glove", help="golve or w2v")
    parser.add_argument('--emb_size', type=int, default=100, help='the dimension of embeddings')

    """ architecture """
    parser.add_argument('--use_ff1', type=str, default="True", help="use ff1")
    parser.add_argument('--use_ff2', type=str, default="True", help="use ff2")
    parser.add_argument("--shortcut", type=str, default="True", help="use shortcut")

    """ training options """
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument("--train_emb", type=str, default="False", help="tune embeddings")
    parser.add_argument("--train_pos_emb", type=str, default="True", help="tune POS embeddings")

    """ MISC """
    parser.add_argument("--trail_id", type=str, default='0', help="trail id")
    parser.add_argument("--arch_id", type=str, default='0', help="trail id")
    parser.add_argument("--max_evals", type=int, default=10, help="max eval #")
    parser.add_argument('--perc', type=int, default=0.0, help='perc')

    argv = parser.parse_args()

    logging.basicConfig(
        filename="logs/hyperopt_" + argv.train_corpus + "_arch_id_" + str(argv.arch_id) + ".log",
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    rs_file = codecs.open("hyeperopt_" + argv.train_corpus + "_" + argv.candidates_num +  "_arch_id_" + str(argv.arch_id) + ".txt", "w")

    rs_file.write("eval id\thidden_size\thidden_size_ffl1\thidden_size_ffl2\tpos_emb_size\tgrad_clip\tword_freq\treg_coef\t"
                  "keep_rate_cell_output\tkeep_rate_input\tkeep_ffl1_rate\tkeep_ffl2_rate\t"
                  "num_params\taverage epoch training\ts@1\ts@2\ts@3\ts@4\ts@1\ts@2\ts@3\ts@4\ts@1\ts@2\ts@3\ts@4\t"
                  "s@1\ts@2\ts@3\ts@4\tepoch\n")
    rs_file.close()

    headers = ["hidden_size", "hidden_size_ffl1", "hidden_size_ffl2", "pos_emb_size", "grad_clip", "word_freq", "reg_coef",
               "keep_rate_cell_output", "keep_rate_input", "keep_ffl1_rate", "keep_ffl2_rate"]
    output_file = open("hyeperopt_" + argv.train_corpus + "_arch_id_" + str(argv.arch_id), 'wb' )
    writer = csv.writer(output_file)
    writer.writerow(headers)

    space = (hp.qloguniform('hidden_size', log(30), log(150), 1),
             hp.qloguniform('hidden_size_ffl1', log(200), log(800), 1),
             hp.qloguniform('hidden_size_ffl2', log(400), log(2000), 1),
             hp.qloguniform('pos_emb_size', log(30), log(100), 1),
             hp.uniform('grad_clip', 1.0, 10.0),
             hp.uniform('word_freq', 1, 8),
             hp.loguniform('reg_coef', log(1e-7), log(1e-2)),
             hp.uniform('keep_rate_cell_output', 0.5, 1.0),
             hp.uniform('keep_rate_input', 0.8, 1.0),
             hp.uniform('keep_ffl1_rate', 0.5, 1.0),
             hp.uniform('keep_ffl2_rate', 0.5, 1.0))

    start_time = time()
    best = fmin(run_wrapper, space, algo=tpe.suggest, max_evals=argv.max_evals)
    end_time = time()

    logging.info("seconds passed: %s" % int(round(end_time-start_time)))
    logging.info("best: %s" % best)



