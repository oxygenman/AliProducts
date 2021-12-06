import albumentations as A
import random
import cv2
from matplotlib import pyplot as plt
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
image = cv2.imread("../../dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = A.Compose([#A.RandomResizedCrop(256,256,scale=(0.8,1.0),p=1.0),
 A.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4,val_shift_limit=0.4, p=0.9),
  A.GaussianBlur(p=0.5),
  A.VerticalFlip(p=0.5),
  A.Transpose(p=0.5),
  A.Normalize(max_pixel_value=1.0, p=1.0),
   A.CoarseDropout(p=0.5, max_width=32, max_height=32),]
)
augmented_image = transform(image=image)['image']

cv2.imwrite("test.jpg",augmented_image)