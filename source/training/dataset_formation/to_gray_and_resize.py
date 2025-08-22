from os import path, makedirs, listdir
import cv2


def process_and_save_images(image_files, dest_dir, category):
    print(category)
    for idx, image_path in enumerate(image_files):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(gray, (48, 48))
        save_path = path.join(dest_dir, category, f'im{idx + 1}.png')
        cv2.imwrite(str(save_path), roi_gray)
        print(f'\rCurrent iteration: {idx}', end="")
    print('\n')


def resize_and_rename_images(input_dir, output_dir):
    makedirs(output_dir, exist_ok=True)
    categories = listdir(input_dir)
    print(categories)

    for category in categories:
        category_input_path = path.join(input_dir, category)
        print(len(listdir(category_input_path)))
        category_output_path = path.join(output_dir, category)

        if path.isdir(category_input_path):
            makedirs(category_output_path, exist_ok=True)
            image_files = [path.join(category_input_path, f) for f in listdir(category_input_path) if
                           path.isfile(path.join(category_input_path, f))]

            print(len(image_files))
            process_and_save_images(image_files, output_dir, category)


if __name__ == "__main__":
    input_directory = 'dataset1'  # Замените на путь к входной папке
    output_directory = 'g_dataset1'  # Замените на путь к выходной папке

    resize_and_rename_images(input_directory, output_directory)
