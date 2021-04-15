import datetime
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

class ClaimDetection:
    """BERT classification model from raw input
    """

    def __init__(self,
                 bert_config_file,
                 bert_ckpt_file,
                 max_seq_len,
                 lr=1e-5):
        """
        bert_config_file: path to bert configuration parameters
        bert_ckpt_file: path to pretrained bert checkpoint
        max_seq_len: maximum sequence lenght
        lr: learning rate
        """
        # create the bert layer
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        output = bert(input_ids)

        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
        # Dropout layer
        cls_out = keras.layers.Dropout(0.8)(cls_out)
        # Dense layer with probibility output
        logits = keras.layers.Dense(units=2, activation="softmax")(cls_out)

        model = keras.Model(inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, max_seq_len))

        # load the pre-trained model weights
        load_stock_weights(bert, bert_ckpt_file)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        self.model = model


    def train(self,
              x_train, y_train,
              x_val, y_val,
              output_dir, max_epoch, patience):
        """
        x_train, y_train: trainable data with supervised labels
        x_val, y_val: validation data
        output_dir: path to save logs
        max_epoch: maximum number of epochs
        patience: number of epochs required to stop training when a monitored metric has stopped improving

        """

        log_dir = os.path.join(output_dir, "log", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        self.model.fit(x=x_train, y=y_train,
                       validation_data=(x_val, y_val),
                       batch_size=16,
                       shuffle=True,
                       epochs=max_epoch,
                       callbacks=[keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
                                  tensorboard_callback])
