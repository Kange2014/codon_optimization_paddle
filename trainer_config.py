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

from sentiment_net import *
from paddle.trainer_config_helpers import *

# whether this config is used for test
is_test = get_config_arg('is_test', bool, False)
# whether this config is used for prediction
is_predict = get_config_arg('is_predict', bool, False)

data_dir = "./data/pre-nature_pe"
dict_dim1, dict_dim2, dict_dim3, class_dim = sentiment_data(data_dir, is_test, is_predict)

################## Algorithm Config #####################

settings(
    batch_size=128,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    model_average=ModelAverage(0.5),
    regularization=L2Regularization(8e-2),
    gradient_clipping_threshold=25)

#################### Network Config ######################
#stacked_lstm_net(
#    dict_dim, class_dim=class_dim, stacked_num=7, is_predict=is_predict)
bidirectional_lstm_net(input_dim1=dict_dim1,input_dim2=dict_dim2,input_dim3=dict_dim3,class_dim=class_dim, fc_size=32, lstm_dim=32, is_predict=is_predict)
#convolution_net(dict_dim1,class_dim=class_dim,hid_dim=64,is_predict=is_predict)
#linear_bidirectional_lstm_net(dict_dim,is_predict=is_predict)
#DB_lstm_net(dict_dim, class_dim=class_dim, depth=4, is_predict=is_predict)
