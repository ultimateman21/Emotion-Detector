from os import path, makedirs, listdir
from shutil import copy


def copy_and_rename_images(src_dirs, dest_dir):
    # Создание выходной директории, если она не существует
    makedirs(dest_dir, exist_ok=True)

    # Проход по каждой входной директории
    for src_dir in src_dirs:
        # Проход по каждой категории в текущей входной директории
        categories = listdir(src_dir)
        for category in categories:
            category_src_path = path.join(src_dir, category)
            category_dest_path = path.join(dest_dir, category)
            print(category)
            iter_ = 0
            if path.isdir(category_src_path):
                # Создание директории для категории в выходной директории, если не существует
                makedirs(category_dest_path, exist_ok=True)

                image_files = [path.join(category_src_path, f) for f in listdir(category_src_path) if
                               path.isfile(path.join(category_src_path, f))]

                for image_file in image_files:
                    new_image_name = f'im{len(listdir(category_dest_path)) + 1}.png'
                    dest_image_path = path.join(category_dest_path, new_image_name)
                    copy(image_file, dest_image_path)
                    iter_ += 1
                    print(f'\rCurrent iteration: {iter_}', end="")
                print('\n')


if __name__ == "__main__":
    input_directory1 = 'archive2/test'  # Замените на путь к первой входной папке
    input_directory2 = 'archive2/train'  # Замените на путь ко второй входной папке
    output_directory = 'dataset2'  # Замените на путь к выходной папке

    # Список входных директорий
    input_directories = [input_directory1, input_directory2]

    # Выполнение копирования и переименования изображений
    copy_and_rename_images(input_directories, output_directory)
