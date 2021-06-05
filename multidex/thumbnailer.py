import os

from fs.osfs import OSFS
import fs.path
from PIL import Image

working_directory = os.path.dirname(os.path.abspath(__file__))

image_fs = OSFS(os.path.join(working_directory, "image_hopper"))
browse_fs = OSFS(os.path.join(working_directory,
                              "plotter/application/assets/browse/mcam/"))
image_type = "jpg"
size = 480, 480

for image_file in image_fs.listdir(""):
    print(image_file)
    fn, ext = fs.path.splitext(image_file)
    input_path = image_fs.getsyspath(image_file)
    output_path = browse_fs.getsyspath(fn + "_browse.jpg")
    image = Image.open(input_path)
    image.thumbnail(size)
    image.save(output_path, "JPEG")
