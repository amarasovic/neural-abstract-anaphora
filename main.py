from data_utils import get_eval_setup
from train import train
from eval import eval
import argparse
import logging
import sys
import codecs
if __name__ == '__main__':
    '''
    1) train the ranking model on the CSN/aritificial data
    2) test the model on the ASN/ARRAU data
        2.1) load the ranking model from the checkpoints
        2.2) get the score for every candidate and rank them
    '''

    # Parameters
    # ==================================================
    parser = argparse.ArgumentParser(description='propositional anaphora resolution', add_help=False,
                                     conflict_handler='resolve')

    parser.add_argument('-mode', default='train', help='train/test')

    """ input """
    parser.add_argument("--eval_setup_id", type=int, default=1, help="eval setup id")
    '''
    eval_setup_id: train, dev, test
    1: artificial small, ASN, ARRAU
    2: csn shell noun (e.g. csn_fact), ARRAU, asn one shell noun (e.g. asn_fact)
    '''
    parser.add_argument('--candidates_num', default="small", help="small = the candidates extracted only from"
                                                                  "        the sent w/ the true antec,"
                                                                  "big_X = small + candidates extracted from"
                                                                  "                 X preceding sentences") # only small used for the paper
    parser.add_argument('--positives_quality', default="all", help="JUST FOR ASN"
                                                                   "all = all true antecedents,"
                                                                   "v1 = w/ confidence greater than "
                                                                   "     the mean of the first two ranked,"
                                                                   "v2 =  w/ confidence greater than"
                                                                   "      the minimum confidence of the first ranked ") # only all used for the paper

    parser.add_argument("--word_freq", type=float, default=10, help="minimum word frequency")

    parser.add_argument('--train_corpus', default=None, help='corpus for training')
    parser.add_argument("--prune_by_len", type=int, default=1, help="threshold for minimum neg. candidate's length")
    parser.add_argument("--prune_by_tag", default="False", help="prune neg. candidates by tag")

    parser.add_argument('--tag_feature', type=str, default="True", help="use tag feature")
    parser.add_argument('--anaphor_feature', type=str, default="True", help="use anaphor feature")
    parser.add_argument('--ctx_feature', type=str, default="True", help="use ctx anaphor feature")
    parser.add_argument("--pretrained_emb", type=str, default="True", help="use pretrained word embeddings")

    parser.add_argument('--emb_type', default="glove", help="golve or w2v")
    parser.add_argument('--emb_size', type=int, default=100, help='the dimension of embeddings')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='the dimension of POS embeddings')

    """ architecture """
    parser.add_argument('--hidden_size', type=int, default=100, help='the dimension of the LSTM hidden state')
    parser.add_argument('--hidden_size_ffl1', type=int, default=400, help='the dimension of the first FF layer')
    parser.add_argument('--hidden_size_ffl2', type=int, default=1024, help='the dimension of the second FF layer')
    parser.add_argument('--use_ff1', type=str, default="True", help="use ff1")
    parser.add_argument('--use_ff2', type=str, default="True", help="use ff2")
    parser.add_argument("--shortcut", type=str, default="True", help="use shortcut")

    """ training options """
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--reg_coef', type=float, default=1e-5, help='L2 reg. rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument("--grad_clip", type=float, default=7.0, help="value gradients are clipped to")
    parser.add_argument("--train_emb", type=str, default="False", help="tune embeddings")
    parser.add_argument("--train_pos_emb", type=str, default="True", help="tune POS embeddings")
    parser.add_argument('--keep_rate_cell_output', type=float, default=0.6, help='keep_rate_cell_output')
    parser.add_argument('--keep_rate_input', type=float, default=0.7, help='keep_rate_input')
    parser.add_argument('--keep_ffl1_rate', type=float, default=0.6, help='keep_ffl1_rate')
    parser.add_argument('--keep_ffl2_rate', type=float, default=0.6, help='keep_ffl2_rate')

    """ test options """
    parser.add_argument('--test_corpus', default=None, help='corpus for testing')
    parser.add_argument('--model', default=None, help='path to model')
    parser.add_argument('--vocab_emb_dict', default=None, help='path to vocab and emb json')
    parser.add_argument('--checkpoint_time', default=None, help='checkpoint time to load from')
    parser.add_argument('--checkpoint_dir', default="/runs/ranking_artificial_v02_small_arch_id_1/1492025627/checkpoints",
                        help='checkpoint dir to load from')

    """ MISC """
    '''arch_id:
    1: the full model
    2: without the context
    3: without the anaphor
    4: without the tag and the shortcut
    5: without the shortcut
    6: without the ffl1
    7: without the ffl2
    '''
    parser.add_argument("--arch_id", type=str, default=None, help="arch id")

    argv = parser.parse_args()

    fname = "results/arrau_results_" +\
            argv.train_corpus + "_" + argv.candidates_num + "_" + '_arch_id_' + argv.arch_id + ".txt"
    arrau_file = codecs.open(fname, "a")

    if argv.mode == "train":
        assert argv.candidates_num
        assert argv.train_corpus
        assert argv.eval_setup_id
        assert argv.arch_id

        logging.basicConfig(
            filename='logs/train_'+ argv.train_corpus + "_" + argv.candidates_num + '_arch_id_' + argv.arch_id + '.log',
            level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

        logging.info('------ARCHITECTURE ID: ' + str(argv.arch_id) + ' ------')
        logging.info('ARCHITECTURE & INPUT SETTINGS')
        logging.info(' candidates number: %s \n use tag feature: %s \n use anaphor feature: %s \n'
                     ' use shortcut: %s \n use FFL1: %s \n use FFL2: %s \n'
                     ' prune by len: %s \n prune by tag: %s \n word freq.: %s \n'
                     ' word emb. type: %s \n word emb. dim: %s \n POS emb. dim: %s' %
                     (argv.candidates_num, argv.tag_feature, argv.anaphor_feature,
                      argv.shortcut, argv.use_ff1, argv.use_ff2,
                      argv.prune_by_len, argv.prune_by_tag, argv.word_freq,
                      argv.emb_type, argv.emb_size, argv.pos_emb_size))

        logging.info('HYPERPARAMETERS')
        logging.info(' LSTM hidden size: %s \n FFL1 size: %s \n FFL2 size: %s \n'
                     ' grad clip: %s \n learning rate: %s \n reg. coef.: %s \n'
                     ' batch size: %s \n # of epochs: %s \n tune emb.: %s \n tune POS emb.: %s' %
                     (argv.hidden_size, argv.hidden_size_ffl1, argv.hidden_size_ffl2,
                      argv.grad_clip, argv.lr, argv.reg_coef,
                      argv.batch_size, argv.num_epoch, argv.train_emb, argv.train_pos_emb))
        logging.info('-----------------------')

        # Data Preparation
        # ==================================================
        train_batches, \
        dev_batches, \
        test_batches,\
        nominal_test_batches,\
        pronominal_test_batches,\
        embeddings, \
        vocabulary, \
        pos_vocabulary = get_eval_setup(argv.eval_setup_id,
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
        parser.add_argument('--vocabulary', default=pos_vocabulary, help="vocabulary")
        parser.add_argument('--embeddings', default=embeddings, help="embeddings matrix")
        parser.add_argument('--train_batches', default=train_batches, help='train batches')
        parser.add_argument('--dev_batches', default=dev_batches, help='dev batches')
        parser.add_argument('--test_batches', default=test_batches, help='test batches')
        parser.add_argument('--nominal_test_batches', default=nominal_test_batches, help='nominal test batches')
        parser.add_argument('--pronominal_test_batches', default=pronominal_test_batches, help='pronominal test batches')
        argv = parser.parse_args()
        precisions, precisions_nominal, precisions_pronominal = train(argv)
        arrau_file.write("\t".join(str(x) for x in precisions) + "\t" +
                         "\t".join(str(x) for x in precisions_nominal) + "\t" +
                         "\t".join(str(x) for x in precisions_pronominal))
        arrau_file.write("\n")

    if argv.mode == "eval":
        assert argv.candidates_num
        assert argv.train_corpus
        assert argv.eval_setup_id
        assert argv.arch_id

        logging.basicConfig(
            filename='logs/train_' + argv.train_corpus + "_" + argv.candidates_num + '_arch_id_' + argv.arch_id + '.log',
            level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

        logging.info('------ARCHITECTURE ID: ' + str(argv.arch_id) + ' ------')
        logging.info('ARCHITECTURE & INPUT SETTINGS')
        logging.info(' candidates number: %s \n use tag feature: %s \n use anaphor feature: %s \n'
                     ' use shortcut: %s \n use FFL1: %s \n use FFL2: %s \n'
                     ' prune by len: %s \n prune by tag: %s \n word freq.: %s \n'
                     ' word emb. type: %s \n word emb. dim: %s \n POS emb. dim: %s' %
                     (argv.candidates_num, argv.tag_feature, argv.anaphor_feature,
                      argv.shortcut, argv.use_ff1, argv.use_ff2,
                      argv.prune_by_len, argv.prune_by_tag, argv.word_freq,
                      argv.emb_type, argv.emb_size, argv.pos_emb_size))

        logging.info('HYPERPARAMETERS')
        logging.info(' LSTM hidden size: %s \n FFL1 size: %s \n FFL2 size: %s \n'
                     ' grad clip: %s \n learning rate: %s \n reg. coef.: %s \n'
                     ' batch size: %s \n # of epochs: %s \n tune emb.: %s \n tune POS emb.: %s' %
                     (argv.hidden_size, argv.hidden_size_ffl1, argv.hidden_size_ffl2,
                      argv.grad_clip, argv.lr, argv.reg_coef,
                      argv.batch_size, argv.num_epoch, argv.train_emb, argv.train_pos_emb))
        logging.info('-----------------------')

        # Data Preparation
        # ==================================================
        train_batches, \
        dev_batches, \
        test_batches, \
        nominal_test_batches, \
        pronominal_test_batches, \
        embeddings, \
        vocabulary, \
        pos_vocabulary = get_eval_setup(argv.eval_setup_id,
                                        argv.word_freq,
                                        argv.train_corpus,
                                        argv.candidates_num,
                                        argv.prune_by_len,
                                        argv.prune_by_tag,
                                        argv.batch_size,
                                        argv.emb_type,
                                        argv.emb_size)
        logging.info("start evaluation...")
        parser.add_argument('--pos_vocabulary', default=pos_vocabulary, help="pos vocabulary")
        parser.add_argument('--vocabulary', default=vocabulary, help="vocabulary")
        parser.add_argument('--embeddings', default=embeddings, help="embeddings matrix")
        parser.add_argument('--train_batches', default=train_batches, help='train batches')
        parser.add_argument('--dev_batches', default=dev_batches, help='dev batches')
        parser.add_argument('--test_batches', default=test_batches, help='test batches')
        parser.add_argument('--nominal_test_batches', default=nominal_test_batches, help='nominal test batches')
        parser.add_argument('--pronominal_test_batches', default=pronominal_test_batches,
                            help='pronominal test batches')
        argv = parser.parse_args()

        eval(argv)

    arrau_file.close()
