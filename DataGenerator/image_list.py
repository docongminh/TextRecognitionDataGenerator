import cv2
import glob

def print_text(file, list_):
    f = open(file, 'w')

    f.writelines(list_)

images = [x.split("/")[-1]+"\n" for x in glob.glob("images/*.jpg")]
print_text("image_list.txt", images)
# for image in images:
#     img = cv2.imread(image, 0)
#
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


