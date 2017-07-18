# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
import numpy as np
from optparse import OptionParser
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import integer_value_sequence
from paddle.trainer.config_parser import parse_config
"""
Usage: run following command to show help message.
  python predict.py -h
"""

codontable = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'X', 'TAG':'X',
    'TGC':'C', 'TGT':'C', 'TGA':'X', 'TGG':'W',
    }

class SentimentPrediction():
    def __init__(self, train_conf, dict_file, model_dir=None, label_file=None):
        """
        train_conf: trainer configure.
        dict_file: word dictionary file name.
        model_dir: directory of model.
        """
        self.train_conf = train_conf
        self.dict_file = dict_file
        self.word_dict = {}
        self.base_dict = {}
        self.aa_dict = {}
        self.dict_dim1, self.dict_dim2,self.dict_dim3 = self.load_dict()
        self.model_dir = model_dir
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.label = None
        if label_file is not None:
            self.load_label(label_file)

        conf = parse_config(train_conf, "is_predict=1")
        self.network = swig_paddle.GradientMachine.createFromConfigProto(
            conf.model_config)
        self.network.loadParameters(self.model_dir)
        input_types = [
            integer_value_sequence(self.dict_dim1), 
            integer_value_sequence(self.dict_dim2),
            integer_value_sequence(self.dict_dim3)
        ]
        self.converter = DataProviderConverter(input_types)

    def load_dict(self):
        """
        Load dictionary from self.dict_file.
        """
        j = 0
        for line_count, line in enumerate(open(self.dict_file, 'r')):
            word = line.strip().split('\t')[0]
            self.word_dict[word] = line_count
            if codontable[word.upper()] not in self.aa_dict:
                 self.aa_dict[codontable[word.upper()]] = j
                 j = j + 1
    
        self.base_dict['g'] = 0
        self.base_dict['a'] = 1
        self.base_dict['c'] = 2
        self.base_dict['t'] = 3

        #for (k,v) in  self.aa_dict.items(): 
        #    print "dict[%s]=" % k,v

        return len(self.base_dict), len(self.word_dict), len(self.aa_dict)

    def load_label(self, label_file):
        """
        Load label.
        """
        self.label = {}
        for v in open(label_file, 'r'):
            self.label[int(v.split('\t')[1])] = v.split('\t')[0]

    def get_index(self, data):
        """
        transform word into integer index according to the dictionary.
        """
        num_base = len(data.strip())
        num_codon = len(data.strip())/3
        bases = [data.strip().lower()[i] for i in range(num_base)]
        words = [data.strip().lower()[i*3:(i*3+3)] for i in range(num_codon)]
        #words = data.strip().split()
        base_slot = [self.base_dict[w] for w in bases if w in self.base_dict]
        word_slot = [self.word_dict[w] for w in words if w in self.word_dict]
        aa_slot = [
                self.aa_dict[codontable[w.upper()]] for w in words if codontable[w.upper()] in self.aa_dict
            ] 
        return base_slot, word_slot, aa_slot

    def batch_predict(self, data_batch):
        input = self.converter(data_batch)
        output = self.network.forwardTest(input)
        prob = output[0]["value"]
        labs = np.argsort(-prob)
        for idx, lab in enumerate(labs):
            if self.label is None:
                print("predicting label is %d" % (lab[0]))
            else:
                print("predicting label is %s" % (self.label[lab[0]]))


def option_parser():
    usage = "python predict.py -n config -w model_dir -d dictionary -i input_file "
    parser = OptionParser(usage="usage: %s [options]" % usage)
    parser.add_option(
        "-n",
        "--tconf",
        action="store",
        dest="train_conf",
        help="network config")
    parser.add_option(
        "-d",
        "--dict",
        action="store",
        dest="dict_file",
        help="dictionary file")
    parser.add_option(
        "-b",
        "--label",
        action="store",
        dest="label",
        default=None,
        help="dictionary file")
    parser.add_option(
        "-c",
        "--batch_size",
        type="int",
        action="store",
        dest="batch_size",
        default=1,
        help="the batch size for prediction")
    parser.add_option(
        "-w",
        "--model",
        action="store",
        dest="model_path",
        default=None,
        help="model path")
    return parser.parse_args()


def main():
    options, args = option_parser()
    train_conf = options.train_conf
    batch_size = options.batch_size
    dict_file = options.dict_file
    model_path = options.model_path
    label = options.label
    swig_paddle.initPaddle("--use_gpu=0")
    predict = SentimentPrediction(train_conf, dict_file, model_path, label)

    batch = []
    for line in sys.stdin:
        bases, words, aas = predict.get_index(line)
        #print("length of words is %d" % len(words))
        #print("length of aas   is %d" % len(aas))
        if words:
            batch.append([bases, words, aas])
            #batch['base'] = bases
            #batch['word'] = words
            #batch['aa'] = aas
        else:
            print('All the words in [%s] are not in the dictionary.' % line)
        if len(batch) == batch_size:
            predict.batch_predict(batch)
            batch = []
    if len(batch) > 0:
        predict.batch_predict(batch)


if __name__ == '__main__':
    main()
