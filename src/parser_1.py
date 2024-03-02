import os
import glob
import random
from xml.etree import ElementTree as ET
from PIL import Image

# Папка с XML-файлами аннотаций
annotations_folder = 'C:/Users/new/Desktop/MO/датасет'
# Папка с изображениями
images_folder = 'C:/Users/new/Desktop/MO/датасет/images'
# Папка назначения для вырезанных кораблей и самолетов
output_folder = 'C:/Users/new/Desktop/MO/репозиторий/project/src/ship_vs_air'

# Создание папки обучающего набора
train_folder = os.path.join(output_folder, 'train')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# Создание папки тестового набора
test_folder = os.path.join(output_folder, 'test')
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Получение списка XML-файлов аннотаций
annotation_files = glob.glob(os.path.join(annotations_folder, '*.xml'))

# Создание функции для вырезания и сохранения изображений
def extract_objects(image_path, objects, output_folder):
    # Открытие изображения
    image = Image.open(image_path)
    
    # Вырезание и сохранение каждого объекта
    for obj in objects:
        # Получение координат ограничивающей рамки
        xmin = float(obj.attrib['xtl'])
        ymin = float(obj.attrib['ytl'])
        xmax = float(obj.attrib['xbr'])
        ymax = float(obj.attrib['ybr'])
        
        # Вырезание объекта
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        
        # Генерация имени файла на основе имени изображения и координат
        file_name = os.path.basename(image_path).split('.')[0] + '_' + str(xmin) + '_' + str(ymin) + '_' + str(xmax) + '_' + str(ymax) + '.png'
        
        # Получение типа объекта
        object_type = obj.attrib['label']
        
        # Папка назначения для объекта
        object_output_folder = os.path.join(output_folder, object_type)
        
        # Создание папки назначения, если она не существует
        if not os.path.exists(object_output_folder):
            os.makedirs(object_output_folder)
        
        # Сохранение вырезанного изображения в формате PNG
        cropped_image.save(os.path.join(object_output_folder, file_name), 'PNG')


# Обработка XML-файлов аннотаций
for file in annotation_files:
    # Парсинг XML-файла аннотации
    root = ET.parse(file).getroot()
    
    # Получение пути к изображению
    images = root.findall('image')
    for image in images:
        image_filename = image.attrib['name']
        image_path = os.path.join(images_folder, image_filename)
        print(image_filename)
        
        # Получение всех объектов на изображении
        objects = image.findall('.//box')
        
        # Генерация случайного числа для определения, относится ли изображение к обучающему или тестовому набору
        random_number = random.uniform(0, 1)
        
        # Определение, в какую папку поместить изображение
        if random_number < 0.8:
            destination_folder = train_folder
        else:
            destination_folder = test_folder
        
        # Вырезание и сохранение объектов
        extract_objects(image_path, objects, destination_folder)
