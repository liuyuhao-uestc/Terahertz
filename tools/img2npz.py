import numpy as np
import glob
import os
from PIL import Image
folder_hq='/Users/liuyuhao/celeba_hq_256'
output_folder='/Users/liuyuhao/celeba_hq_npy'
png_files = glob.glob(os.path.join(folder_hq, '*.jpg'))
for png_file in png_files:
    img=Image.open(png_file).convert('RGB')
    img_array=np.array(img)
    base_name=os.path.splitext(os.path.basename(png_file))[0]
    npy_file=os.path.join(output_folder, 'imgHQ' + base_name + '.npy')
    np.save(npy_file,img_array)
