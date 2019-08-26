import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.utils import to_categorical, np_utils
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def keras_learning(X_train, X_test, y_train, y_test):
    # モデルの構築
    model = Sequential()
    model.add(Dense(input_dim=4, output_dim=100, activation='relu'))
    model.add(Dense(input_dim=100, output_dim=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # 訓練の開始
    csv_logger = CSVLogger('log.csv', separator=',', append=False)
    history = model.fit(X_train, y_train,
                        batch_size=32, epochs=100,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[csv_logger])

    return model

def keras_predict(X_train, X_test, y_train, y_test, model):
    # モデルの評価(evaluate)
    train_score = model.evaluate(X_train, y_train)
    test_score = model.evaluate(X_test, y_test)
    print('Train score:', train_score[0])
    print('Train accuracy:', train_score[1])
    print('Test score:', test_score[0])
    print('Test accuracy:', test_score[1])

    # モデルの評価(predict)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_train = np.argmax(pred_train, axis=1) # for one-hot
    pred_test = np.argmax(pred_test, axis=1) # for one-hot
    print(pred_train)
    print(np.argmax(y_train, axis=1))
    print(pred_test)
    print(np.argmax(y_test, axis=1))

def save_model(model):
    # モデルの保存
    model_json = model.to_json()
    with open('model.json', 'w') as file:
        file.write(model_json)
    model.save('weights.hdf5')

def load_model():
    # モデルの読み込み
    with open('model.json', 'r') as file:
        model_json = file.read()
        model = model_from_json(model_json)
    model.load_weights('weights.hdf5')

    return model

def main():
    # データセットをロード
    iris = load_iris()
    print(iris.data)
    print(iris.feature_names)
    print(iris.target)
    print(iris.target_names)
    print(type(iris.data))
    print(type(iris.target))
    
    data_X = iris.data
    data_y = iris.target
    data_y = np_utils.to_categorical(data_y)
    print(data_X)
    print(data_y)

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    model = keras_learning(X_train, X_test, y_train, y_test)
    keras_predict(X_train, X_test, y_train, y_test, model)
    save_model(model)
    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    keras_predict(X_train, X_test, y_train, y_test, model)

if __name__ == '__main__':
    main()