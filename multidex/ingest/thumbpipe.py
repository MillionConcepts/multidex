import io

from PIL import Image
from marslab.composition import Composition


# make consistently-sized thumbnails out of the asdf context images. we
# might eventually want a type of output earmarked for this, or to write the
# online thumbnails out locally. (something like this can be inserted as a
# send into look pipelines...?)


def remove_alpha(image):
    return image.convert("RGB")


def pil_crop(image, bounds):
    bounds = (
        bounds[0],
        bounds[3],
        image.size[0] - bounds[1],
        image.size[1] - bounds[2],
    )
    cropped = image.crop(bounds)
    return cropped


def thumber(image, scale):
    image.thumbnail((image.size[0] / scale, image.size[1] / scale))
    return image


def jpeg_buffer(image):
    buffer = io.BytesIO()
    image.save(buffer, "jpeg")
    buffer.seek(0)
    return buffer


def default_thumbnailer():
    params = {"crop": {"bounds": (230, 185, 155, 170)}, "thumb": {"scale": 3}}
    steps = {
        "load": Image.open,
        "flatten": remove_alpha,
        "crop": pil_crop,
        "thumb": thumber,
        "write": jpeg_buffer,
    }
    return Composition(steps=steps, parameters=params)
