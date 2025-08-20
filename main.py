from PyQt6.QtWidgets import QApplication, QWidget, QGroupBox, QLabel, QTextEdit, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from numpy import clip, array
from sys import argv, exit
from PIL import Image
from os import remove
import cv2


class LoadModel(QThread):
    model_loader = pyqtSignal(object)

    def run(self):
        model = load_model('source/emotion_model.hdf5', compile=False)
        model.make_predict_function()
        self.model_loader.emit(model)


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Emotion Detector')
        self.setWindowIcon(QIcon('source/icon.png'))
        self.setGeometry(500, 30, 0, 0)

        layout = QVBoxLayout(self)

        media_box = QGroupBox('Мультимедиа')
        media_layout = QVBoxLayout()

        self.media_label = QLabel()
        media_layout.addWidget(self.media_label)

        media_box.setLayout(media_layout)
        layout.addWidget(media_box)

        self.edit = QTextEdit()
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.edit.setReadOnly(True)
        layout.addWidget(self.edit)

        control_box = QGroupBox('')
        self.control_layout = QHBoxLayout()

        self.foto_button = QPushButton('Сделать фото')
        self.foto_button.clicked.connect(self.capture_photo)
        self.control_layout.addWidget(self.foto_button)

        self.own_button = QPushButton('Загрузить своё')
        self.own_button.clicked.connect(self.load_file)
        self.control_layout.addWidget(self.own_button)

        self.back_button = QPushButton('Вернуться к камере')
        self.back_button.clicked.connect(self.back_action)
        self.back_button.hide()

        self.define_button = QPushButton('Определить')
        self.define_button.clicked.connect(self.define)
        self.define_button.setEnabled(False)

        control_box.setLayout(self.control_layout)
        layout.addWidget(control_box)

        self.setLayout(layout)
        self.frame = None

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_frame)

        self.cap = cv2.VideoCapture(0)
        self.video_timer.start(40)

        self.model = None
        self.load_model_thread = LoadModel()
        self.load_model_thread.model_loader.connect(self.model_loaded)
        self.load_model_thread.start()

    def model_loaded(self, model):
        self.model = model
        self.define_button.setEnabled(True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = QImage(frame, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_BGR888)
            self.media_label.setPixmap(QPixmap.fromImage(image))

    def change_content(self, purpose):
        for i in reversed(range(self.control_layout.count())):
            wid = self.control_layout.itemAt(i).widget()
            wid.hide()
            self.control_layout.removeWidget(wid)
        if purpose == 'photo':
            self.video_timer.stop()
            self.cap.release()
            self.control_layout.addWidget(self.back_button)
            self.back_button.show()
            self.control_layout.addWidget(self.define_button)
            self.define_button.show()
            self.minimize_window()
        elif purpose == 'video':
            self.cap = cv2.VideoCapture(0)
            self.video_timer.start(40)
            self.edit.clear()
            self.control_layout.addWidget(self.foto_button)
            self.foto_button.show()
            self.control_layout.addWidget(self.own_button)
            self.own_button.show()
            self.minimize_window()

    def capture_photo(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            image = QImage(frame, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_BGR888)
            self.media_label.setPixmap(QPixmap.fromImage(image))
        self.change_content('photo')

    def load_file(self):
        self.cap.release()
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Photo', '', 'Images (*.png *.xpm *.jpg *.jpeg)')
        if filename:
            self.change_content('photo')
            with Image.open(filename) as img:
                img.save('source/1.png', 'PNG')
            frame = cv2.imread('source/1.png')
            height = QApplication.primaryScreen().geometry().height() - 250
            if frame.shape[0] > height:
                aspect_ratio = height / frame.shape[0]
                frame = cv2.resize(frame, (int(frame.shape[1] * aspect_ratio), height))
            image = QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_BGR888)
            self.frame = frame
            self.media_label.setPixmap(QPixmap.fromImage(image))
        else:
            self.change_content('video')
        self.minimize_window()

    def define(self):
        try:
            emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
            size = self.model.input_shape[1:3]
            img = self.frame

            mtcnn = MTCNN()
            faces = [self.define_square(x["box"]) for x in mtcnn.detect_faces(img)]

            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = cv2.mean(img_g[img_g.shape[:2][0] - 2: img_g.shape[:2][0], 0:img_g.shape[:2][1]])[0]
            padded_img = cv2.copyMakeBorder(img_g, top=40, bottom=40, left=40, right=40, borderType=cv2.BORDER_CONSTANT, value=[mean] * 3)
            faces1 = [cv2.resize(padded_img[max(0, i[2]): i[3], max(0, i[0]): i[1]], size).astype('float32') / 255.0 - 0.5 * 2.0 for i in faces]
            pred = self.model(array(faces1))

            emotions = [dict(box=faces[i], emotions={emotion_labels[j]: round(float(s), 2) for j, s in enumerate(f)}) for i, f in enumerate(pred)]

            out = []
            for idx, face in enumerate(emotions):
                box = face['box']
                top_emotions = sorted(face['emotions'].items(), key=lambda item: item[1], reverse=True)[:3]

                cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2)

                scale = round(abs(box[0] - box[1]) / 100)

                cv2.putText(img, str(idx), (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3 + (0.1 * scale), (0, 255, 255), 1)

                y_off = box[2] + box[3] - box[2] + 20
                for emotion, score in top_emotions:
                    label = f'{emotion}: {score * 100:.2f}%'
                    cv2.putText(img, label, (box[0], y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.3 + (0.1 * scale), (255, 0, 0), 1)
                    y_off += 5 + (5 * scale)
                out.append(f"{idx}: {'  '.join([f'{e} - {s * 100:.2f}%' for e, s in top_emotions])}")

            self.define_button.hide()
            image = QImage(img, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format.Format_BGR888)
            self.media_label.setPixmap(QPixmap.fromImage(image))
            self.edit.setText('\n'.join(out))
        except Exception as e:
            self.edit.setText(str(e))

    @staticmethod
    def define_square(bbox):
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff

        x1, x2, y1, y2 = x - 40, x + w, y - 40, y + h
        x1 = clip(x1 + 40, a_min=0, a_max=None)
        y1 = clip(y1 + 40, a_min=0, a_max=None)
        return x1, x2 + 20, y1, y2 + 20

    def back_action(self):
        self.change_content('video')

    def minimize_window(self):
        self.setMinimumSize(0, 0)
        self.resize(0, 0)
        self.move(500, 0)

    def closeEvent(self, event):
        self.cap.release()
        remove('source/1.png')
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(argv)
    window = CameraApp()
    window.show()
    exit(app.exec())
