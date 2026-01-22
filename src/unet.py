from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(
        n_filters, 3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    conv = Conv2D(
        n_filters, 3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal"
    )(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    
    up = Conv2DTranspose(
                 n_filters, 
                 3,
                 strides=(2, 2),
                 padding="same")(expansive_input)
    
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                 3,
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(merge)
    conv = Conv2D(n_filters,
                 3,
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(conv)
    
    return conv