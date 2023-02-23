import os
import re
from pathlib import Path

import wand.image

import plotter.application


def make_svg_font_style_block():
    asset_path = str(
        Path(Path(plotter.application.__file__).parent, "assets/")
    )
    font_path = Path(asset_path, "fonts")
    font_stylesheet_path = Path(asset_path, "css/fonts.css")
    with open(font_stylesheet_path) as font_stylesheet:
        fonts_css = font_stylesheet.read()
    fonts_css = fonts_css.replace("../fonts", str(font_path))
    return f"<style>{fonts_css}</style>"


def inject_style_into_svg(svgtext, style):
    """
    crudely inject a style block (or whatever) into the first <defs> block
    of an svg file expressed as a string
    """
    defs_end = re.search("defs.*?>", svgtext).span()[1]
    return svgtext[:defs_end] + style + svgtext[defs_end:]


def load_svg_as_wand(injected_svg):
    # have to do this weird hack to trick ImageMagick into finding the fonts...
    with open("temp_svg.svg", "w") as file:
        file.write(injected_svg)
    wand_image = wand.image.Image(filename="temp_svg.svg")
    os.unlink("temp_svg.svg")
    return wand_image


def inject_fonts_and_reload(svgtext):
    font_style_block = make_svg_font_style_block()
    svgtext = inject_style_into_svg(svgtext, font_style_block)
    return load_svg_as_wand(svgtext)
