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
from paddle.trainer.PyDataProvider2 import *


def hook(settings, dictionary1,dictionary2, dictionary3, **kwargs):
    settings.base_dict = dictionary1
    settings.word_dict = dictionary2
    settings.aa_dict = dictionary3
    settings.input_types = {
       "base": integer_value_sequence(len(settings.base_dict)), 
       "word": integer_value_sequence(len(settings.word_dict)),
        "aa": integer_value_sequence(len(settings.aa_dict)),
       "label": integer_value(6)
       #dense_vector(1) # linear
    }
    settings.logger.info('base dict len : %d' % (len(settings.base_dict)))
    settings.logger.info('word dict len : %d' % (len(settings.word_dict)))
    settings.logger.info('aa dict len : %d' % (len(settings.aa_dict)))

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

@provider(init_hook=hook)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            label, comment = line.strip().split('\t\t')
            label = int(label)
            #label = float(label)
            
            num_base = len(comment)
            num_codon = len(comment)/3
            bases = [comment.lower()[i] for i in range(num_base)]
            words = [comment.lower()[i*3:(i*3+3)] for i in range(num_codon)]
            #words = comment.split()
            base_slot = [
                settings.base_dict[w] for w in bases if w in settings.base_dict
            ]
            word_slot = [
                settings.word_dict[w] for w in words if w in settings.word_dict
            ]
            aa_slot = [
                settings.aa_dict[codontable[w.upper()]] for w in words if codontable[w.upper()] in settings.aa_dict
            ]            
            if not base_slot:
                continue
            if not word_slot:
                continue
            if not aa_slot:
                continue
            yield {"base":base_slot, "word":word_slot, "aa": aa_slot,"label":label}
