from os import path, makedirs, listdir
from shutil import copy2


def copy_and_number_images(src_dir, dest_dir, start_index=0):
    if not path.exists(dest_dir):
        makedirs(dest_dir)

    index = start_index
    for filename in listdir(src_dir):
        src_file = path.join(src_dir, filename)
        if path.isfile(src_file):
            dest_file = path.join(dest_dir, f'im{index}.png')
            copy2(src_file, dest_file)
            index += 1
            print(f'\rCurrent iteration: {index}', end="")

    return index


def merge_directories(input_dir1, input_dir2, output_dir):
    if not path.exists(output_dir):
        makedirs(output_dir)

    # Создаем словарь для отслеживания текущего индекса для каждой категории
    category_indices = {}

    # Обработка первой входной директории
    for category in listdir(input_dir1):
        print(category)
        src_category_dir = path.join(input_dir1, category)
        dest_category_dir = path.join(output_dir, category)
        if path.isdir(src_category_dir):
            category_indices[category] = copy_and_number_images(src_category_dir, dest_category_dir)
        print('\n')

    # Обработка второй входной директории
    for category in listdir(input_dir2):
        print(category)
        src_category_dir = path.join(input_dir2, category)
        dest_category_dir = path.join(output_dir, category)
        if path.isdir(src_category_dir):
            if category in category_indices:
                category_indices[category] = copy_and_number_images(src_category_dir, dest_category_dir, category_indices[category])
            else:
                category_indices[category] = copy_and_number_images(src_category_dir, dest_category_dir)
        print('\n')


if __name__ == "__main__":
    input_dir1 = 'dataset2'
    input_dir2 = 'g_dataset1'
    output_dir = 'm_dataset'

    # Пример использования
    merge_directories(input_dir1, input_dir2, output_dir)
