#!/usr/bin/env python

'''Demonstrate Python wrapper of C apriltag library by running on camera frames.'''

from argparse import ArgumentParser
import cv2
import numpy as np
import apriltag

# for some reason pylint complains about members being undefined :(
# pylint: disable=E1101
class AprilLabeler:

    def __init__(self, image_list, folder):
        parser = ArgumentParser(
            description='test apriltag Python bindings')

        parser.add_argument('device_or_movie', metavar='INPUT', nargs='?', default=0,
                            help='Movie to load or integer ID of camera device')

        apriltag.add_arguments(parser)

        options = parser.parse_args()
        detector = apriltag.Detector(options)

        content = None
        with open(image_list) as f:
            content = f.readlines()
            print(content[0])


        self.content = content
        self.detector = detector
        self.index = 0
        self.total_images = len(content)
        self.folder = folder

    def collect_labeled_data(self, N):
        # N is the number of labeled data wanted
        image_list = []
        labels = []
        num_in_each = []
        for i in range(N):
            idx = self.index

            im = self.content[idx].split()
            print(im)

            frame = cv2.imread(self.folder + im[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detections, dimg = self.detector.detect(gray, return_image=True)

            num_detections = len(detections)
            print 'Detected {} tags.\n'.format(num_detections)

            label_per_detection = []
            for j, detection in enumerate(detections):
                print 'Detection {} of {}:'.format(j+1, num_detections)
                print
                # print detection.tostring(indent=2)
                print
                print(type(detection))
                dic = {}
                dic["Family"] = detection[0]
                dic["ID"] = detection[1]
                dic["Hamming error"] = detection[2]
                dic["Goodness"] = detection[3]
                dic["Decision margin"] = detection[4]
                dic["Homography"] = detection[5]
                dic["Center"] = detection[6]
                dic["Corners"] = detection[7]
                label_per_detection.append(dic)
                print(dic)

            num_in_each.append(num_detections)
            image_list.append(gray)
            labels.append(label_per_detection)




            self.index = (self.index + 1) % self.total_images
        return image_list, labels, num_in_each


def main():
    '''Main function.'''
    folder = "example_images/"
    img_file = "image_list.txt"
    al = AprilLabeler(img_file, folder)
    imgs, labels, nums = al.collect_labeled_data(10)
    print("returned labels", labels)
    print("number of tags per image", nums)
    for idx, img in enumerate(imgs):
        for n in range(nums[idx]):
            label_n = labels[idx][n]
            corners = np.array(label_n["Corners"])
            print("Corners")
            print(corners)
            minx = int(np.floor(min(corners[:,0])))
            miny = int(np.floor(min(corners[:,1])))
            maxx = int(np.ceil(max(corners[:,0])))
            maxy = int(np.ceil(max(corners[:,1])))
            shifted_corners = corners - np.array([minx, miny])
            print("Shifted Corners")
            print(shifted_corners)
            crop_img = img[miny:maxy, minx:maxx]
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
