# -*- coding: utf-8 -*-

import numpy as np
import cv2
from os.path import dirname, join, basename
from glob import glob

num=0
for fn in glob(join(dirname(__file__)+'other', '*.jpg')):
    img = cv2.imread(fn)
    res = cv2.resize(img,(64,128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(r'./photos/'+str(num)+'.jpg', res)
    num += 1

print "all done"

cv2.waitKey(0)
cv2.destroyAllWindows()
