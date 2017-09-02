
from keras.models import Sequential
from keras.layers import Dense, Dropout
from commons import Utilities
from keras.utils import np_utils, generic_utils
from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

class KerasWrapper():
    _model = None
    _param=None

    def __init__(self, logFile=None, utilObj=None):

        if (utilObj != None):
            self._util = utilObj
        elif (logFile != None):
            self._util = Utilities.Utility()
            self._util.setupLogFileLoc(logFile)

        self._util.logDebug('KerasWrapper','Keras will only be initialised upon training')

    def save(self, filename):
        self._model.save(filename)

    def load(self,filename):
        from keras.models import load_model
        # Use Keras loading model
        self._model = load_model(filename)

    def fit(self, Xtrain, Ytrain):
        # create model
        print('Xtrain.shape[1]:',Xtrain.shape[1])
        self._model= Sequential()
        self._model.add(Dense(32, input_dim=Xtrain.shape[1],activation='relu'))
        self._model.add(Dense(8, activation='relu'))
        # self._model.add(Dropout(0.2))
        self._model.add(Dense(8, activation='relu'))
        self._model.add(Dense(8, activation='relu'))
        # self._model.add(Dropout(0.1))
        self._model.add(Dense(8, activation='relu'))
        self._model.add(Dense(8, activation='relu'))
        self._model.add(Dense(8, activation='relu'))
        self._model.add(Dense(8, activation='relu'))
        # self._model.add(Dropout(0.2))
        self._model.add(Dense(4, activation='softmax'))
        # Compile model
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        Ytrain = to_categorical(Ytrain, 4)
        print('Shape X:',Xtrain.shape)
        print('Shape Y:', Ytrain.shape)
        self._model.fit(Xtrain,Ytrain,epochs=200, batch_size=10, verbose=1)
        # from keras.utils.vis_utils import plot_model
        # plot_model(self._model, to_file='/home/kah1/model.png')

    def predict_proba(self,Xtest):
        self._util.logError('KerasWrapper','PredictProba is not supported...returning zeros')
        return [0]*len(Xtest)

    def predict(self,Xtest):
        prediction=self._model.predict_classes(Xtest)
        return prediction
        pass

