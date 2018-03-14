import stain_utils as utils
import stainNorm_Macenko

import cv2

i3 = utils.read_image('./data/i3.png')
i2 = utils.read_image('./data/i2.png')
n = stainNorm_Macenko.Normalizer()
n.fit(i2)
out = n.transform(i3)
cv2.imwrite('ans1.jpg',out)

