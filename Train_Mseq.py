import numpy as np
import os 
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def custom_split(features_main,silences,labels):
    idx_new = np.random.permutation(features_main.shape[0])
    features_main,silences,labels = features_main[idx_new], silences[idx_new], labels[idx_new]
    
    split = int(len(features_main)*0.7)
    X_train_feat = features_main[:split]
    X_test_feat = features_main[split:]

    X_train_silence = silences[:split]
    X_test_silence = silences[split:]

    Y_train = labels[:split]
    Y_test = labels[split:]

    return X_train_feat,X_test_feat,X_train_silence,X_test_silence,Y_train,Y_test

def get_model():
    dtype = 'float64'
    dropout=0.2

    input_data_1 = Input(name='main_Features', shape=X_train_feat[0].shape, dtype=dtype)
    input_data_2 = Input(name='silence', shape=(1), dtype=dtype)

    att_in = Bidirectional(LSTM(64,return_sequences=True,kernel_regularizer=l2(),name='lstm1'))(input_data_1)
    att_out = Bidirectional(LSTM(64,return_sequences=False,name='lstm2'))(att_in)


    concatted = Concatenate()([att_out,input_data_2])
    x = Dense(units=128, activation='tanh', name='fc')(concatted)
    x = Dropout(dropout, name='dropout_2')(x)

    y_pred = Dense(units=Y_test.shape[1], activation='softmax', name='softmax')(x) 

    K.clear_session()
    model = Model(inputs=[input_data_1,input_data_2], outputs=y_pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model 


if __name__ == '__main__':
    combined_main_features = np.load('data/Traces_Mseq/new_features_main.npy')
    combined_silences = np.load('data/Traces_Mseq//new_silences.npy')
    combined_labels = np.load('data/Traces_Mseq//new_labels.npy')

    ohe = OneHotEncoder(sparse=False)
    labels_u = combined_labels.reshape(-1,1)
    labels_ohe = ohe.fit_transform(labels_u)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    K.clear_session()
    np.random.seed(20)
    
    X_train_feat,X_test_feat,X_train_silence,X_test_silence,Y_train,Y_test = custom_split(combined_main_features,combined_silences,labels_ohe)

    adam = Adam()
    early = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=10)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                            patience=3,
                                            verbose=1, mode='min', min_delta=0.001, cooldown=3,
                                            min_lr=1e-9) 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    tf.keras.backend.set_floatx('float64')   

    checkpoint = ModelCheckpoint('Models/M_seq_fingeprinting', monitor='val_loss', verbose=0,
                                    save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=16)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                            patience=5,
                                            verbose=1, mode='min', min_delta=0.0001, cooldown=3,
                                            min_lr=1e-9)  

    model = get_model()
    model.fit([X_train_feat,X_train_silence], Y_train,
                        batch_size=128, epochs=100,
                        validation_data=([X_test_feat,X_test_silence], Y_test),
                        callbacks=[early, reduceLROnPlat,checkpoint])