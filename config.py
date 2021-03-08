from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dropout
FEATURE_DIM = 37
CROSSVALID_EPOCH = 10
EPOCH = 200
LR = 1e-4
MAX_LEN = 8
STEPS_PER_EPOCH = 2000
BATCH_SIZE = 16
VALIDATION_WINDOWS_SIZE = 5


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def bulid_model():
    # inputs = keras.Input(shape=(None, FEATURE_DIM))
    # x = layers.Masking(
    #     mask_value=0., input_shape=(None, FEATURE_DIM))(inputs)
    # x = TransformerBlock(FEATURE_DIM, 8, 32, rate=0.3)(x)
    # x = TransformerBlock(FEATURE_DIM, 8, 32, rate=0.3)(x)
    # # x = TransformerBlock(FEATURE_DIM, 8, 32, rate=0.3)(x)
    # # x = TransformerBlock(FEATURE_DIM, 8, 32, rate=0.3)(x)
    # x = layers.GlobalMaxPooling1D()(x)
    # x = layers.Dropout(0.3)(x)
    # x = layers.Dense(16, activation='elu')(x)
    # x = layers.Dropout(0.3)(x)
    # outputs = layers.Dense(3, activation='softmax')(x)
    # model = keras.Model(inputs, outputs, name="transformer")
    inputs = keras.Input(shape=(None, FEATURE_DIM))
    x = layers.Masking(mask_value=-1., input_shape=(None, FEATURE_DIM))(inputs)
    x = layers.Bidirectional(layers.LSTM(
        32, return_sequences=True, input_shape=(None, FEATURE_DIM)), input_shape=(None, FEATURE_DIM))(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(20, activation='elu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='elu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="LSTM_Attention")
    # model = keras.Sequential()
    # model.add(layers.Bidirectional(layers.LSTM(
    #     32, return_sequences=True, input_shape=(None, FEATURE_DIM)), input_shape=(None, FEATURE_DIM)))
    # model.add(layers.MultiHeadAttention(num_heads=2, key_dim=32))
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(64, activation='elu'))
    # model.add(layers.LayerNormalization())
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(10, activation='elu'))
    # model.add(layers.LayerNormalization())
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    return model


def build_pretraining_model():
    inputs = keras.Input(shape=(None, FEATURE_DIM))
    x = TransformerBlock(FEATURE_DIM, 8, 64, rate=0.3)(inputs)
    x = TransformerBlock(FEATURE_DIM, 8, 64, rate=0.3)(x)
    x = TransformerBlock(FEATURE_DIM, 8, 64, rate=0.3)(x)
    x = TransformerBlock(FEATURE_DIM, 8, 64, rate=0.3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(FEATURE_DIM, activation='elu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(FEATURE_DIM)(x)
    model = keras.Model(inputs, outputs, name="pretrain")
    # model = keras.Sequential()
    # model.add(layers.Bidirectional(layers.LSTM(
    #     32, return_sequences=True, input_shape=(None, FEATURE_DIM)), input_shape=(None, FEATURE_DIM)))
    # model.add(layers.MultiHeadAttention(num_heads=2, key_dim=32))
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(64, activation='elu'))
    # model.add(layers.LayerNormalization())
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(10, activation='elu'))
    # model.add(layers.LayerNormalization())
    # model.add(Dropout(0.2))
    # model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    return model
