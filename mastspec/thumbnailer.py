from fs.osfs import OSFS
import fs.path
from PIL import Image

image_fs = OSFS('./static_in_pro/img/roi_source')
browse_fs = OSFS('./static_in_pro/img/roi_browse')
image_type = 'jpg'
size = 480, 480

for image_file in image_fs.listdir(''):
    print(image_file)
    fn, ext = fs.path.splitext(image_file)
    input_path = image_fs.getsyspath(image_file)
    output_path = browse_fs.getsyspath(fn + '_browse.jpg')
    image = Image.open(input_path)
    image.thumbnail(size)
    image.save(output_path, "JPEG")