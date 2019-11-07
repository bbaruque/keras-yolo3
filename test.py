from yolo import YOLO
from PIL import Image

FLAGS = {'model_path': '', 
        'anchors_path': '' , 
        'classes_path': ''}

yolov3 = YOLO(**vars(FLAGS))

test_file = '../raccoon_dataset-master/data/raccoon_labels_modif_test.csv'

def detect_img(yolo,img_path):
    try:
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()

with open(test_file) as fp:
   line = fp.readline()
   cnt = 1
   while line:
       print("Line {}: {}".format(cnt, line.strip()))

       elements = line.strip().split(' ')
       print("File path: {}".format(elements[0]))

       detect_img(yolov3,elements[0])

       line = fp.readline()
       cnt += 1

yolov3.close_session()