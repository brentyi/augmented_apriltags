import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from argparse import ArgumentParser

from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2


import matplotlib.pyplot as plt
import random

######################################################################################
folder = "high/"
image_list = "high_files.txt"
out_folder = "high_out/"

def show_cv2(img):
    # swap b and r channels
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])

#     plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

def rand_color():
    def component():
        return random.randint(0,255)
    return (component(), component(), component())

def compute_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def line_intersection(p1, p2):
    L1 = compute_line(p1[:2], p1[2:4])
    L2 = compute_line(p2[:2], p2[2:4])

    d  = L1[0] * L2[1] - L1[1] * L2[0]
    dx = L1[2] * L2[1] - L1[1] * L2[2]
    dy = L1[0] * L2[2] - L1[2] * L2[0]
    if d != 0:
        x = dx / d
        y = dy / d
    else:
        # assume negative coordinates are invalid
        x = -1
        y = -1
    return np.asarray((x,y))



cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'six_pm_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


#################

#################
# In[10]:
#####
def find_corners(img):
  img_height, img_width, _ = img.shape
  img_diag = int(np.sqrt(img_height ** 2 + img_width ** 2))
  # show_cv2(img)
  img_size = np.linalg.norm(img.shape)

  img_blur = img
  blur_size = 1 + 2 * int(img_size / 100)
  if blur_size > 1:
      blur_size = (blur_size, blur_size)
      img_blur = cv2.GaussianBlur(img, blur_size, 0)
  img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
  # print(img_hsv[0,0])

  #img = cv2.resize(img,(500,500))
  thresh = np.zeros(img.shape[:2], dtype=np.uint8)
  thresh[img_hsv[:,:,2] <= 160] = 255
  thresh[img_hsv[:,:,1] >= 127] = 0



  # plt.imshow(thresh)
  # plt.show()

  #Create default parametrization LSD
  lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

  #Detect lines in the image
  lines = lsd.detect(thresh)[0] #Position 0 of the returned tuple are the detected lines
  #Draw detected lines in the image
  #img = lsd.drawSegments(img,lines)
  valid_lines = []
  for line in lines:
      x1, y1, x2, y2 = line[0]
      length = np.linalg.norm([x1 - x2, y1 - y2])
      if length < img_diag / 10:
          continue

      avg_x = (x1 + x2) / 2
      avg_y = (y1 + y2) / 2

      m_x = img_width / 2 - avg_x
      m_y = img_height / 2 - avg_y
      m_x /= np.linalg.norm([m_x, m_y])
      m_y /= np.linalg.norm([m_x, m_y])

      def get_v(l):
          return img_hsv[int(avg_y + m_y * l), int(avg_x + m_x * l), 2]
      if get_v(5) > get_v(-5):
          continue
      valid_lines.append(line[0])

  valid_lines.sort(key=lambda l: -(l[0] ** 2 + l[1] ** 2))

  points = []
  def vote_point(new_p, weight):
      for i in range(len(points)):
          p = points[i][0]
          old_weight = points[i][1]
          if np.linalg.norm(p - new_p) < 20:
              p *= old_weight / (weight + old_weight)
              p += new_p * weight / (weight + old_weight)
              points[i][1] += weight
              return
      points.append([new_p, weight])

  for line_index, line in enumerate(valid_lines):
      x1, y1, x2, y2 = line
      p1 = np.asarray([x1, y1])
      p2 = np.asarray([x2, y2])

      slope = p2 - p1
      length = np.linalg.norm(slope)
      slope /= length
      #angle = np.arctan2(diff[1], diff[0])


      vote_point(p1, length)
      vote_point(p2, length)
  #     cv2.line(img, tuple(p1), tuple(p2), rand_color(), 20)

      for other_line in valid_lines[line_index + 1:]:
          if line is other_line:
              continue

          ox1, oy1, ox2, oy2 = other_line
          op1 = np.asarray([ox1, oy1])
          op2 = np.asarray([ox2, oy2])

          oslope = op2 - op1
          olength = np.linalg.norm(oslope)
          oslope /= olength
          #oangle = np.arctan2(odiff[1], odiff[0])

          intersection = line_intersection(line, other_line)
          if 0 <= intersection[0] < img_width and 0 <= intersection[1] <= img_height and np.abs(slope @ oslope) < 0.5:
              vote_point(intersection, max([length, olength]))

  done = False
  while done == False:
      points.sort(key=lambda p: -p[1])
      final_points = points[:4]
      final_points.sort(key=lambda p: np.arctan2(p[0][1] - img_height / 2, p[0][0] - img_width / 2))

      done = True
      for i in range(4):
          v1 = (final_points[i][0] - final_points[i - 1][0]).astype(np.float32)
          v2 = (final_points[i][0] - final_points[(i + 1) % len(final_points)][0]).astype(np.float32)
          v1 /= np.linalg.norm(v1)
          v2 /= np.linalg.norm(v2)
          theta = np.arccos(v1 @ v2)
          # print(theta)
          if theta < np.pi / 20 or theta > np.pi * 19 / 20:
              # print("xxx")
              done = False
              # a bit hacky
              final_points[i][-1] = -1

  final_points = np.asarray([p[0] for p in final_points], dtype=np.int32)
  print(final_points)
  cv2.drawContours(img, [final_points], 0, (0,0,255), 8)



content = None
with open(image_list) as f:
    content = f.readlines()
    print(content[0])
frames = []
for i in range(len(content)):
    idx = i
    im = content[idx].split()
    print(im)
    image_path = folder + im[0]
    image = Image.open(image_path)
    frame = load_image_into_numpy_array(image)

    frames.append(frame)

  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

idx = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for f in frames:
      # faster_rcnn
      # ret, image_np = cap.read()
      image_np = f
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      thresh = 0.9
      num = 0
      for s in scores[0,:]:
        if s > 0.9:
          num += 1
        else:
          break

      W, H, C = image_np.shape

      for n in range(num):

        # print(box)
        # print(boxes[0][n])
        minx = int(np.floor(H * boxes[0][n][1]))
        miny = int(np.floor(W * boxes[0][n][0]))
        maxx = int(np.ceil(H * boxes[0][n][3]))
        maxy = int(np.ceil(W * boxes[0][n][2]))

        top = (minx, miny)
        bot = (maxx, maxy)
        cv2.rectangle(image_np, top, bot, (0,255,0), 4)
        d = 35

        miny_d = max(miny-d, 0)
        maxy_d = min(maxy+d, W-1)
        minx_d = max(minx-d, 0)
        maxx_d = min(maxx+d, H-1)

        crop_img = image_np[miny_d:maxy_d, minx_d:maxx_d]
        cv2.rectangle(image_np, (minx_d, miny_d), (maxx_d, maxy_d), (255,255,0), 3)
        try:
          find_corners(crop_img)
        except:
          pass

      dest = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

      cv2.imwrite(out_folder+ '%05d.png'%idx,dest)
      idx +=1
      ###

      ## displaying image

      # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      # if cv2.waitKey(25) & 0xFF == ord('q'):
        # cv2.destroyAllWindows()
        # break

