from sklearn.model_selection import train_test_split
from os import path, makedirs, listdir
import cv2


# Функция для обработки изображений
def process_and_save_images(image_files, dest_dir, category):
    print(category)
    for idx, image_path in enumerate(image_files):
        image = cv2.imread(image_path)
        save_path = path.join(dest_dir, category, path.basename(image_path))
        cv2.imwrite(str(save_path), cv2.resize(image, (64, 64)))
        print(f'\rCurrent iteration: {idx}', end="")
    print('\n')


dataset_dir = 'm_dataset'

# Пути для обучающей и проверочной директорий
train_dir = 'train_data'
val_dir = 'val_data'

# Создание директорий, если они не существуют
makedirs(train_dir, exist_ok=True)
makedirs(val_dir, exist_ok=True)

# Получение списка категорий
categories = listdir(dataset_dir)

# Деление на обучающую и проверочную выборки для каждой категории
for category in categories:
    category_dir = path.join(dataset_dir, category)
    if path.isdir(category_dir):
        # Создание директорий для текущей категории, если они не существуют
        makedirs(path.join(train_dir, category), exist_ok=True)
        makedirs(path.join(val_dir, category), exist_ok=True)

        # Получение списка изображений в текущей категории
        image_files = [path.join(category_dir, f) for f in listdir(category_dir)
                       if path.isfile(path.join(category_dir, f))]
        # Деление на обучающую и проверочную выборки
        train_files, val_files = train_test_split(image_files, test_size=0.08, random_state=42)

        # Обработка и сохранение изображений в соответствующие директории
        process_and_save_images(train_files, train_dir, category)
        process_and_save_images(val_files, val_dir, category)
