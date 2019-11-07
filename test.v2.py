import argparse

from yolo import YOLO
from PIL import Image

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

test_file = '../../DataSets/labels/HeadsTesting_modif.csv'
predictions_file = '../../DataSets/detections/heads_labels_predicted_tinyYolo_adaptedAnchors.csv'

def center(box):
    print(box)
    x_cent = round((box[2]-box[0])/2)
    y_cent = round((box[3]-box[1])/2)
    return (x_cent, y_cent)

def detect_img(yolo,img_path):
    try:
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
    else:

        response = yolo.detect_image(image)
        print(response)

        r_image = response[0]
        out_boxes = response[1]
        out_scores = response[2]
        out_classes = response[3]
        elapsed = response[4]

        #r_image.show()
        print('boxes: {} | scores:{} | classes:{}'.format(str(out_boxes), str(out_scores), str(out_classes)))
        return out_boxes, out_scores, out_classes, elapsed

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    yolov3 = YOLO(**vars(FLAGS))

    with open(test_file) as fp:
        with open(predictions_file, 'w') as fp_pred:
           line = fp.readline()
           cnt = 1
           while line:
               #print("Line {}: {}".format(cnt, line.strip()))

               elements = line.strip().split(' ')
               print("File path: {}".format(elements[0]))

               response = detect_img(yolov3,elements[0])
               print(response)

               if len(response[1]) == 0: # if it doesn't recognize any head
                   result = '{}\n'.format(elements[0])
                   fp_pred.write(result)

               else: # if it does recognize one or more heads
                   out_boxes = response[0]
                   out_scores = response[1]
                   out_classes = response[2]
                   elapsed = response[3]

                   for b, s, c in zip(out_boxes, out_scores, out_classes):
                       result = '{};{};{};{};{};{}\n'.format(elements[0], b, s, c, center(b), elapsed)
                       print(result)
                       fp_pred.write(result)

               line = fp.readline()
               cnt += 1

    yolov3.close_session()

    print('Test finished correctly')