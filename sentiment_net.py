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

from os.path import join as join_path

from paddle.trainer_config_helpers import *


def sentiment_data(data_dir=None,
                   is_test=False,
                   is_predict=False,
                   train_list="train.list",
                   test_list="test.list",
                   dict_file="dict.txt"):
    """
    Predefined data provider for sentiment analysis.
    is_test: whether this config is used for test.
    is_predict: whether this config is used for prediction.
    train_list: text file name, containing a list of training set.
    test_list: text file name, containing a list of testing set.
    dict_file: text file name, containing dictionary.
    """
    class_dim = len(open(join_path(data_dir, 'labels.list')).readlines())
    
    if data_dir is not None:
        train_list = join_path(data_dir, train_list)
        test_list = join_path(data_dir, test_list)
        dict_file = join_path(data_dir, dict_file)

    train_list = train_list if not is_test else None
    
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

    word_dict = dict()
    aa_dict = dict()
    base_dict = dict()
    j = 0
    with open(dict_file, 'r') as f:
        for i, line in enumerate(open(dict_file, 'r')):
            word = line.split('\t')[0]
            word_dict[word] = i
            if codontable[word.upper()] not in aa_dict:
                 aa_dict[codontable[word.upper()]] = j
                 j = j + 1
    
    base_dict['g'] = 0
    base_dict['a'] = 1
    base_dict['c'] = 2
    base_dict['t'] = 3

    define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={'dictionary1': base_dict,'dictionary2': word_dict, 'dictionary3': aa_dict})

    return len(base_dict), len(word_dict), len(aa_dict), class_dim


def bidirectional_lstm_net(input_dim1,
                           input_dim2,
                           input_dim3,
                           class_dim=2,
                           emb_dim=128,
                           fc_size=4,
                           lstm_dim=128,
                           is_predict=False):
    data1 = data_layer("base", input_dim1)
    emb1 = embedding_layer(input=data1, size=emb_dim)
    
    data2 = data_layer("word", input_dim2)
    emb2 = embedding_layer(input=data2, size=emb_dim)

    data3 = data_layer("aa", input_dim3)
    emb3 = embedding_layer(input=data3, size=emb_dim)
    
    ## nt cnn + aa cnn + codon bi-lstm + fc layer
    layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    #conv_1 = sequence_conv_pool(
    #    input=emb1, context_len=1, hidden_size=fc_size)
    #dropout1 = dropout_layer(input=conv_1, dropout_rate=0.5)
    #inputs = [conv_1]

    #conv_3 = sequence_conv_pool(
    #    input=emb3, context_len=1, hidden_size=fc_size)
    #dropout3 = dropout_layer(input=conv_3, dropout_rate=0.5)
    #inputs.append(conv_3)
    
    #conv_3 = sequence_conv_pool(
    #    input=emb1, context_len=3, hidden_size=fc_size)
    #dropout3 = dropout_layer(input=conv_3, dropout_rate=0.5)
    #inputs.append(dropout3)

    #conv = sequence_conv_pool(
    #    input=emb2, context_len=1, hidden_size=fc_size)
    #dropout1 = dropout_layer(input=conv_1, dropout_rate=0.5)
    #inputs.append(conv)

    #fc1 = fc_layer(input = inputs, size = fc_size, layer_attr = ExtraLayerAttribute(drop_rate=0.5))

    #for i in range(0,4):
    #    conv = sequence_conv_pool(input=emb2, context_len=1, hidden_size=fc_size)
    #    dropout2 = dropout_layer(input=conv_7, dropout_rate=0.5)
    #    inputs.append(conv)
    
    #conv_5 = sequence_conv_pool(
    #    input=emb1, context_len=5, hidden_size=lstm_dim)
    #dropout3 = dropout_layer(input=conv_5, dropout_rate=0.5)
    #inputs.append(dropout3)
    
    #cnn_output = fc_layer(input=inputs,size=fc_size*2, act=ReluActivation())
    #for i in range(3,10,2):
    #    conv = sequence_conv_pool(input=emb2, context_len=3, hidden_size=lstm_dim)
    #    inputs.append(conv)

    #
    #for i in range(0,1):
    #    conv_15 = sequence_conv_pool(input=emb, context_len=15, hidden_size=lstm_dim)
    #    inputs.append(conv_15)

    #conv_41 = sequence_conv_pool(
    #    input=emb, context_len=41, hidden_size=lstm_dim)
    #dropout4 = dropout_layer(input=conv_41, dropout_rate=0.5)
    #inputs.append(dropout4)
    #for i in range(0,2):
    #    conv_41 = sequence_conv_pool(input=dropout1, context_len=41, hidden_size=lstm_dim)
    #    inputs.append(conv_41)

    #fc1 = fc_layer(input=dropout1,size=lstm_dim,act=LinearActivation())

    #bi_lstm2 = bidirectional_lstm(input=fc1, size=lstm_dim)
    #dropout2 = dropout_layer(input=bi_lstm2, dropout_rate=0.5)
    #output = fc_layer(input=dropout2, size=class_dim, act=SoftmaxActivation())
    
    bi_lstm1 = bidirectional_lstm(input=emb1, size=lstm_dim, return_seq=True)
    dropout1 = dropout_layer(input=bi_lstm1, dropout_rate=0.5)

    bi_lstm2 = bidirectional_lstm(input=emb2, size=lstm_dim, return_seq=True)
    dropout2 = dropout_layer(input=bi_lstm2, dropout_rate=0.5)
    
    bi_lstm3 = bidirectional_lstm(input=emb3, size=lstm_dim, return_seq=True)
    dropout3 = dropout_layer(input=bi_lstm3, dropout_rate=0.5)
    
    conv_1 = sequence_conv_pool(
        input=dropout1, context_len=13, hidden_size=fc_size)
    inputs = [conv_1]

    conv_2 = sequence_conv_pool(
        input=dropout2, context_len=3, hidden_size=fc_size)
    inputs.append(conv_2)

    conv_3 = sequence_conv_pool(
        input=dropout3, context_len=3, hidden_size=fc_size)
    inputs.append(conv_3)

    #conv_3 = sequence_conv_pool(
    #    input=dropout1, context_len=3, hidden_size=fc_size)
    #inputs = [conv_3]
    #inputs.append(conv_3)

    #conv_5 = sequence_conv_pool(
    #    input=dropout1, context_len=5, hidden_size=fc_size)

    #inputs.append(conv_5)

    #conv_7 = sequence_conv_pool(
    #    input=dropout1, context_len=7, hidden_size=fc_size)

    #inputs.append(conv_7)

    #layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    #cnn1 = fc_layer(input=inputs, size=fc_size, act=ReluActivation(),layer_attr=layer_attr)
    
    #conv_4 = sequence_conv_pool(
    #    input=inputs, context_len=4, hidden_size=fc_size)

    #layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    #bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
    #lstm1 = lstmemory(
    #    input=emb, act=ReluActivation(), bias_attr=bias_attr, layer_attr=layer_attr)
    #lstm_last = pooling_layer(input=lstm1, pooling_type=MaxPooling())

    #output = fc_layer(input=inputs, size=1, act=LinearActivation())

    #if not is_predict:
    #    lbl = data_layer("label", 1)
    #    outputs(mse_cost(input=output, label=lbl))

    fc = fc_layer(input = inputs, #[conv_1,conv_2,conv_3], 
                  size = fc_size, layer_attr = ExtraLayerAttribute(drop_rate=0.5),act=ReluActivation())

    output = fc_layer(input=fc,
                             size=class_dim,
                             act=SoftmaxActivation())

    if not is_predict:
        lbl = data_layer("label", 1)
        outputs(classification_cost(input=output, label=lbl))
    else:
        outputs(output)


def stacked_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3,
                     is_predict=False):
    """
    A Wrapper for sentiment classification task.
    This network uses bi-directional recurrent network,
    consisting three LSTM layers. This configure is referred to
    the paper as following url, but use fewer layrs.
        http://www.aclweb.org/anthology/P15-1109

    input_dim: here is word dictionary dimension.
    class_dim: number of categories.
    emb_dim: dimension of word embedding.
    hid_dim: dimension of hidden layer.
    stacked_num: number of stacked lstm-hidden layer.
    is_predict: is predicting or not.
                Some layers is not needed in network when predicting.
    """
    hid_lr = 1e-3
    assert stacked_num % 2 == 1

    layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    fc_para_attr = ParameterAttribute(learning_rate=hid_lr)
    lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
    relu = ReluActivation()
    linear = LinearActivation()

    data = data_layer("word", input_dim)
    emb = embedding_layer(input=data, size=emb_dim)

    fc1 = fc_layer(input=emb, size=hid_dim, act=linear, bias_attr=bias_attr)
    lstm1 = lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr, layer_attr=layer_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = fc_layer(
            input=inputs,
            size=hid_dim,
            act=linear,
            param_attr=para_attr,
            bias_attr=bias_attr)
        lstm = lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr,
            layer_attr=layer_attr)
        inputs = [fc, lstm]

    fc_last = pooling_layer(input=inputs[0], pooling_type=MaxPooling())
    lstm_last = pooling_layer(input=inputs[1], pooling_type=MaxPooling())
    output = fc_layer(
        input=[fc_last, lstm_last],
        size=class_dim,
        act=SoftmaxActivation(),
        bias_attr=bias_attr,
        param_attr=para_attr)

    if is_predict:
        outputs(output)
    else:
        outputs(classification_cost(input=output, label=data_layer('label', 1)))

def one_dimension_conv_pool(input,
                       context_len,
                       hidden_size,
                       name=None,
                       context_start=None,
                       pool_type=None,
                       context_proj_layer_name=None,
                       context_proj_param_attr=False,
                       fc_layer_name=None,
                       fc_param_attr=None,
                       fc_bias_attr=None,
                       fc_act=None,
                       pool_bias_attr=None,
                       fc_attr=None,
                       context_attr=None,
                       pool_attr=None):
    """
    Text convolution pooling layers helper.
    Text input => Context Projection => FC Layer => Pooling => Output.
    :param name: name of output layer(pooling layer name)
    :type name: basestring
    :param input: name of input layer
    :type input: LayerOutput
    :param context_len: context projection length. See
                        context_projection's document.
    :type context_len: int
    :param hidden_size: FC Layer size.
    :type hidden_size: int
    :param context_start: context projection length. See
                          context_projection's context_start.
    :type context_start: int or None
    :param pool_type: pooling layer type. See pooling_layer's document.
    :type pool_type: BasePoolingType.
    :param context_proj_layer_name: context projection layer name.
                                    None if user don't care.
    :type context_proj_layer_name: basestring
    :param context_proj_param_attr: context projection parameter attribute.
                                    None if user don't care.
    :type context_proj_param_attr: ParameterAttribute or None.
    :param fc_layer_name: fc layer name. None if user don't care.
    :type fc_layer_name: basestring
    :param fc_param_attr: fc layer parameter attribute. None if user don't care.
    :type fc_param_attr: ParameterAttribute or None
    :param fc_bias_attr: fc bias parameter attribute. False if no bias,
                         None if user don't care.
    :type fc_bias_attr: ParameterAttribute or None
    :param fc_act: fc layer activation type. None means tanh
    :type fc_act: BaseActivation
    :param pool_bias_attr: pooling layer bias attr. None if don't care.
                           False if no bias.
    :type pool_bias_attr: ParameterAttribute or None.
    :param fc_attr: fc layer extra attribute.
    :type fc_attr: ExtraLayerAttribute
    :param context_attr: context projection layer extra attribute.
    :type context_attr: ExtraLayerAttribute
    :param pool_attr: pooling layer extra attribute.
    :type pool_attr: ExtraLayerAttribute
    :return: output layer name.
    :rtype: LayerOutput
    """
    # Set Default Value to param
    context_proj_layer_name = "%s_conv_proj" % name \
        if context_proj_layer_name is None else context_proj_layer_name

    with mixed_layer(
            name=context_proj_layer_name,
            size=context_len,
            act=LinearActivation(),
            layer_attr=context_attr) as m:
        m += context_projection(
            input,
            context_len=context_len,
            context_start=context_start,
            padding_attr=context_proj_param_attr)

    fc_layer_name = "%s_conv_fc" % name \
        if fc_layer_name is None else fc_layer_name
    print(m.size)
    fl = fc_layer(
        name=fc_layer_name,
        input=m,
        size=hidden_size,
        act=fc_act,
        layer_attr=fc_attr,
        param_attr=fc_param_attr,
        bias_attr=fc_bias_attr)

    return pooling_layer(
        name=name,
        input=fl,
        pooling_type=pool_type,
        bias_attr=pool_bias_attr,
        layer_attr=pool_attr)

def convolution_net(input_dim,
                    class_dim=2,
                    emb_dim=128,
                    hid_dim=128,
                    is_predict=False):
    data = data_layer("base",input_dim)
    emb = embedding_layer(input=data, size=emb_dim)
    con_project = context_projection(emb,context_len=13)
    print(con_project)
    conv1 = sequence_conv_pool(
            input=emb, context_len=13, hidden_size=hid_dim, fc_act=ReluActivation())
    dropout1 = dropout_layer(input=conv1, dropout_rate=0.15)
    print(dropout1.size)
    
    #data = Layer(inputs=dropout1,size=1)
    #fc = fc_layer(input=dropout1,size=hid_dim,act=LinearActivation())
    layer_attr = ExtraLayerAttribute(drop_rate=0.15)
    conv2 = one_dimension_conv_pool(input=dropout1, context_len=13, hidden_size=hid_dim)
    #dropout2 = dropout_layer(input=conv2, dropout_rate=0.15)
    #conv2 = img_conv_layer(input=data,filter_size=1,filter_size_y=1,num_filters=hid_dim,num_channels=1,act=ReluActivation(),layer_attr=layer_attr)

    #conv3 = sequence_conv_pool(
    #        input=dropout2, context_len=13, hidden_size=hid_dim)

    #dropout3 = dropout_layer(input=conv3, dropout_rate=0.15)
    

    #fc1 = fc_layer(input=dropout1,size=hid_dim,act=ReluActivation(),layer_attr=layer_attr)
    #fc2 = fc_layer(input=fc1,size=hid_dim,act=ReluActivation(),layer_attr=layer_attr)

    fc = fc_layer(input=conv2,size=64,act=ReluActivation())
    output = fc_layer(input=fc,
                             size=class_dim,
                             act=SoftmaxActivation())
    if not is_predict:
        lbl = data_layer("label", 1)
        cost = classification_cost(input=output, label=lbl)
        outputs(cost)
    else:
        outputs(output)

def linear_bidirectional_lstm_net(input_dim,
                           emb_dim=128,
                           lstm_dim=128,
                           is_predict=False):
    data = data_layer("word", input_dim)
    emb = embedding_layer(input=data, size=emb_dim)
    bi_lstm = bidirectional_lstm(input=emb, size=lstm_dim)
    dropout = dropout_layer(input=bi_lstm, dropout_rate=0.5)
    output = fc_layer(input=dropout, size=1, act=LinearActivation())

    if not is_predict:
        lbl = data_layer("label", 1)
        outputs(mse_cost(input=output, label=lbl))
    else:
        outputs(output)

def DB_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     depth=2,
                     is_predict=False):
    """
    A Wrapper for sentiment classification task.
    This network uses bi-directional recurrent network,
    consisting three LSTM layers. This configure is referred to
    the paper as following url, but use fewer layrs.
        http://www.aclweb.org/anthology/P15-1109

    input_dim: here is word dictionary dimension.
    class_dim: number of categories.
    emb_dim: dimension of word embedding.
    hid_dim: dimension of hidden layer.
    depth: number of deep bi-lstm hidden layer.
    is_predict: is predicting or not.
                Some layers is not needed in network when predicting.
    """
    hid_lr = 1e-3
    assert depth % 2 == 0

    layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    fc_para_attr = ParameterAttribute(learning_rate=hid_lr)
    lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
    relu = ReluActivation()
    linear = LinearActivation()

    data = data_layer("word", input_dim)
    emb = embedding_layer(input=data, size=emb_dim)

    fc1 = fc_layer(input=emb, size=hid_dim, act=linear, bias_attr=bias_attr)
    lstm1 = lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr, layer_attr=layer_attr)

    inputs = [fc1, lstm1]
    for i in range(1, depth):
        fc = mixed_layer(
            input=[
            full_matrix_projection(
                input=inputs[0], param_attr=fc_para_attr),
            full_matrix_projection(
                input=inputs[1], param_attr=lstm_para_attr)
            ],
            size=hid_dim,
            act=linear,
            bias_attr=bias_attr)
        lstm = lstmemory(
            input=fc,
            reverse=(i % 2) == 1,
            act=relu,
            bias_attr=bias_attr,
            layer_attr=layer_attr)
        inputs = [fc, lstm]

    #fc_last = pooling_layer(input=inputs[0], pooling_type=MaxPooling())
    #lstm_last = pooling_layer(input=inputs[1], pooling_type=MaxPooling())
    output = fc_layer(
        #input=[fc_last, lstm],
        input=inputs[1],
        size=class_dim,
        act=SoftmaxActivation(),
        bias_attr=bias_attr)

    if is_predict:
        outputs(output)
    else:
        outputs(classification_cost(input=output, label=data_layer('label', 1)))