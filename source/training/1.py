from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, SeparableConv2D, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from numpy import array, arange, random, argmax, unique, sum
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from seaborn import set, heatmap
import matplotlib.pyplot as plt
from pandas import DataFrame
from os import path, listdir
import json
import cv2


# Функция для загрузки данных
def load_data(data_dir):
    X, Y = [], []
    emotion_s = listdir(data_dir)

    emotions_p = 'emotions.json'
    with open(emotions_p, 'w') as file:
        json.dump(emotion_s, file)

    for emotion_idx, emotion in enumerate(emotion_s):
        emotion_dir = path.join(data_dir, emotion)
        print(emotion)
        for idx, image_name in enumerate(listdir(emotion_dir)):
            image_path = path.join(emotion_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            X.append(image)
            Y.append(emotion_idx)
            print(f'\rCurrent iteration: {idx}', end="")
        print('\n')
    X = array(X)
    Y = to_categorical(Y, len(emotion_s))
    indices = arange(len(X))
    random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    return X, Y, emotion_s


# Создание модели
def create_model(len_classes):
    input_layer = Input(shape=(64, 64, 1))

    # Начальный блок
    x = Conv2D(8, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Первый разветвлённый блок
    x1 = SeparableConv2D(16, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = SeparableConv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Второй разветвлённый блок
    x1 = SeparableConv2D(32, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Третий разветвлённый блок
    x1 = SeparableConv2D(64, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Четвёртый разветвлённый блок
    x1 = SeparableConv2D(128, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Финальный блок
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(len_classes, activation='softmax')(x)

    model_ = Model(inputs=input_layer, outputs=output_layer)
    model_.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_


# Функция для сохранения или загрузки модели
def model_save(model_p, history_p, model_, x_train, y_train):
    if path.exists(model_p):
        model_ = load_model(model_p)
        with open(history_p, 'r') as file:
            history = json.load(file)
        return model_, history
    else:
        y_decode = argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', classes=unique(y_decode), y=y_decode)
        print(class_weights)
        history = model_.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.1, class_weight=dict(enumerate(class_weights)))
        history = history.history
        model_.save(model_p)
        with open(history_p, 'w') as file:
            json.dump(history, file)
        return model_, history


# Загружаем данные
train_data_dir = 'train_data'
val_data_dir = 'val_data'
X_train, Y_train, emotions = load_data(train_data_dir)

X_val, Y_val, _ = load_data(val_data_dir)

model_filename = "emotion_model.hdf5"
history_filename = "saved_history_4.json"

model = create_model(len(emotions))
model_1, history_1 = model_save(model_filename, history_filename, model, X_train, Y_train)

# Проверка модели
val_loss, val_accuracy = model_1.evaluate(X_val, Y_val)
print(f'Validation Loss: {val_loss}\nValidation Accuracy: {val_accuracy}')

# Соотношение верно предсказанных значений
y_pred = model_1.predict(X_val)
print(y_pred)
y_pred_classes = argmax(y_pred, axis=1)
y_true = argmax(Y_val, axis=1)
print(f'Correct Predictions: {sum(y_pred_classes == y_true)}\nTotal Predictions: {len(y_true)}')

cm_data = confusion_matrix(y_true, y_pred_classes)
cm = DataFrame(cm_data, columns=emotions, index=emotions)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize=(10, 5))
plt.title('Confusion Matrix', fontsize=20)
set(font_scale=1.2)
ax = heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
plt.show()
