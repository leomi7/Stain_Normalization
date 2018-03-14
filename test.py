import stain_utils as utils
import stainNorm_Reinhard

import cv2

i1 = utils.read_image('./data/i1.png')
i2 = utils.read_image('./data/i2.png'
)
n = stainNorm_Reinhard.Normalizer()
n.fit(i1)
out = n.transform(i2)
cv2.imwrite('ans.jpg',out)

