"""
possible placeholder module giving options for fixed/solid colors in graph
right now just a selection of named CSS colors that aren't too light to be
pointless
"""
SOLID_MARKER_COLORS = (
    {"label": "aqua", "value": "aqua"},
    {"label": "aquamarine", "value": "aquamarine"},
    {"label": "azure", "value": "azure"},
    {"label": "black", "value": "black"},
    {"label": "blue", "value": "blue"},
    {"label": "blueviolet", "value": "blueviolet"},
    {"label": "brown", "value": "brown"},
    {"label": "burlywood", "value": "burlywood"},
    {"label": "cadetblue", "value": "cadetblue"},
    {"label": "chartreuse", "value": "chartreuse"},
    {"label": "chocolate", "value": "chocolate"},
    {"label": "coral", "value": "coral"},
    {"label": "cornflowerblue", "value": "cornflowerblue"},
    {"label": "crimson", "value": "crimson"},
    {"label": "cyan", "value": "cyan"},
    {"label": "darkblue", "value": "darkblue"},
    {"label": "darkcyan", "value": "darkcyan"},
    {"label": "darkgoldenrod", "value": "darkgoldenrod"},
    {"label": "darkgray", "value": "darkgray"},
    {"label": "darkgreen", "value": "darkgreen"},
    {"label": "darkkhaki", "value": "darkkhaki"},
    {"label": "darkmagenta", "value": "darkmagenta"},
    {"label": "darkolivegreen", "value": "darkolivegreen"},
    {"label": "darkorange", "value": "darkorange"},
    {"label": "darkorchid", "value": "darkorchid"},
    {"label": "darkred", "value": "darkred"},
    {"label": "darksalmon", "value": "darksalmon"},
    {"label": "darkseagreen", "value": "darkseagreen"},
    {"label": "darkslateblue", "value": "darkslateblue"},
    {"label": "darkslategray", "value": "darkslategray"},
    {"label": "darkturquoise", "value": "darkturquoise"},
    {"label": "darkviolet", "value": "darkviolet"},
    {"label": "deeppink", "value": "deeppink"},
    {"label": "deepskyblue", "value": "deepskyblue"},
    {"label": "dimgray", "value": "dimgray"},
    {"label": "dodgerblue", "value": "dodgerblue"},
    {"label": "firebrick", "value": "firebrick"},
    {"label": "forestgreen", "value": "forestgreen"},
    {"label": "fuchsia", "value": "fuchsia"},
    {"label": "gainsboro", "value": "gainsboro"},
    {"label": "gold", "value": "gold"},
    {"label": "goldenrod", "value": "goldenrod"},
    {"label": "gray", "value": "gray"},
    {"label": "green", "value": "green"},
    {"label": "greenyellow", "value": "greenyellow"},
    {"label": "honeydew", "value": "honeydew"},
    {"label": "hotpink", "value": "hotpink"},
    {"label": "indigo", "value": "indigo"},
    {"label": "khaki", "value": "khaki"},
    {"label": "lavender", "value": "lavender"},
    {"label": "lawngreen", "value": "lawngreen"},
    {"label": "lightblue", "value": "lightblue"},
    {"label": "lightcoral", "value": "lightcoral"},
    {"label": "lightcyan", "value": "lightcyan"},
    {"label": "lightgoldenrodyellow", "value": "lightgoldenrodyellow"},
    {"label": "lightgreen", "value": "lightgreen"},
    {"label": "lightpink", "value": "lightpink"},
    {"label": "lightsalmon", "value": "lightsalmon"},
    {"label": "lightseagreen", "value": "lightseagreen"},
    {"label": "lightskyblue", "value": "lightskyblue"},
    {"label": "lightslategray", "value": "lightslategray"},
    {"label": "lightsteelblue", "value": "lightsteelblue"},
    {"label": "lightyellow", "value": "lightyellow"},
    {"label": "lime", "value": "lime"},
    {"label": "limegreen", "value": "limegreen"},
    {"label": "magenta", "value": "magenta"},
    {"label": "maroon", "value": "maroon"},
    {"label": "mediumaquamarine", "value": "mediumaquamarine"},
    {"label": "mediumblue", "value": "mediumblue"},
    {"label": "mediumorchid", "value": "mediumorchid"},
    {"label": "mediumpurple", "value": "mediumpurple"},
    {"label": "mediumseagreen", "value": "mediumseagreen"},
    {"label": "mediumslateblue", "value": "mediumslateblue"},
    {"label": "mediumspringgreen", "value": "mediumspringgreen"},
    {"label": "mediumturquoise", "value": "mediumturquoise"},
    {"label": "mediumvioletred", "value": "mediumvioletred"},
    {"label": "midnightblue", "value": "midnightblue"},
    {"label": "mistyrose", "value": "mistyrose"},
    {"label": "navy", "value": "navy"},
    {"label": "olive", "value": "olive"},
    {"label": "olivedrab", "value": "olivedrab"},
    {"label": "orange", "value": "orange"},
    {"label": "orangered", "value": "orangered"},
    {"label": "orchid", "value": "orchid"},
    {"label": "palegreen", "value": "palegreen"},
    {"label": "paleturquoise", "value": "paleturquoise"},
    {"label": "palevioletred", "value": "palevioletred"},
    {"label": "papayawhip", "value": "papayawhip"},
    {"label": "peru", "value": "peru"},
    {"label": "pink", "value": "pink"},
    {"label": "plum", "value": "plum"},
    {"label": "powderblue", "value": "powderblue"},
    {"label": "purple", "value": "purple"},
    {"label": "red", "value": "red"},
    {"label": "rosybrown", "value": "rosybrown"},
    {"label": "royalblue", "value": "royalblue"},
    {"label": "rebeccapurple", "value": "rebeccapurple"},
    {"label": "saddlebrown", "value": "saddlebrown"},
    {"label": "salmon", "value": "salmon"},
    {"label": "sandybrown", "value": "sandybrown"},
    {"label": "seagreen", "value": "seagreen"},
    {"label": "sienna", "value": "sienna"},
    {"label": "skyblue", "value": "skyblue"},
    {"label": "slateblue", "value": "slateblue"},
    {"label": "slategray", "value": "slategray"},
    {"label": "springgreen", "value": "springgreen"},
    {"label": "steelblue", "value": "steelblue"},
    {"label": "tan", "value": "tan"},
    {"label": "teal", "value": "teal"},
    {"label": "thistle", "value": "thistle"},
    {"label": "tomato", "value": "tomato"},
    {"label": "turquoise", "value": "turquoise"},
    {"label": "violet", "value": "violet"},
    {"label": "wheat", "value": "wheat"},
    {"label": "yellow", "value": "yellow"},
    {"label": "yellowgreen", "value": "yellowgreen"},
)

MARKER_SYMBOLS = (
    {"label": "circle", "value": "circle"},
    {"label": "circle-open", "value": "circle-open"},
    {"label": "circle-dot", "value": "circle-dot"},
    {"label": "circle-open-dot", "value": "circle-open-dot"},
    {"label": "square", "value": "square"},
    {"label": "square-open", "value": "square-open"},
    {"label": "square-dot", "value": "square-dot"},
    {"label": "square-open-dot", "value": "square-open-dot"},
    {"label": "diamond", "value": "diamond"},
    {"label": "diamond-open", "value": "diamond-open"},
    {"label": "diamond-dot", "value": "diamond-dot"},
    {"label": "diamond-open-dot", "value": "diamond-open-dot"},
    {"label": "cross", "value": "cross"},
    {"label": "cross-open", "value": "cross-open"},
    {"label": "cross-dot", "value": "cross-dot"},
    {"label": "cross-open-dot", "value": "cross-open-dot"},
    {"label": "x", "value": "x"},
    {"label": "x-open", "value": "x-open"},
    {"label": "x-dot", "value": "x-dot"},
    {"label": "x-open-dot", "value": "x-open-dot"},
    {"label": "triangle-up", "value": "triangle-up"},
    {"label": "triangle-up-open", "value": "triangle-up-open"},
    {"label": "triangle-up-dot", "value": "triangle-up-dot"},
    {"label": "triangle-up-open-dot", "value": "triangle-up-open-dot"},
    {"label": "triangle-down", "value": "triangle-down"},
    {"label": "triangle-down-open", "value": "triangle-down-open"},
    {"label": "triangle-down-dot", "value": "triangle-down-dot"},
    {"label": "triangle-down-open-dot", "value": "triangle-down-open-dot"},
    {"label": "triangle-left", "value": "triangle-left"},
    {"label": "triangle-left-open", "value": "triangle-left-open"},
    {"label": "triangle-left-dot", "value": "triangle-left-dot"},
    {"label": "triangle-left-open-dot", "value": "triangle-left-open-dot"},
    {"label": "triangle-right", "value": "triangle-right"},
    {"label": "triangle-right-open", "value": "triangle-right-open"},
    {"label": "triangle-right-dot", "value": "triangle-right-dot"},
    {
        "label": "triangle-right-open-dot",
        "value": "triangle-right-open-dot",
    },
    {"label": "triangle-ne", "value": "triangle-ne"},
    {"label": "triangle-ne-open", "value": "triangle-ne-open"},
    {"label": "triangle-ne-dot", "value": "triangle-ne-dot"},
    {"label": "triangle-ne-open-dot", "value": "triangle-ne-open-dot"},
    {"label": "triangle-se", "value": "triangle-se"},
    {"label": "triangle-se-open", "value": "triangle-se-open"},
    {"label": "triangle-se-dot", "value": "triangle-se-dot"},
    {"label": "triangle-se-open-dot", "value": "triangle-se-open-dot"},
    {"label": "triangle-sw", "value": "triangle-sw"},
    {"label": "triangle-sw-open", "value": "triangle-sw-open"},
    {"label": "triangle-sw-dot", "value": "triangle-sw-dot"},
    {"label": "triangle-sw-open-dot", "value": "triangle-sw-open-dot"},
    {"label": "triangle-nw", "value": "triangle-nw"},
    {"label": "triangle-nw-open", "value": "triangle-nw-open"},
    {"label": "triangle-nw-dot", "value": "triangle-nw-dot"},
    {"label": "triangle-nw-open-dot", "value": "triangle-nw-open-dot"},
    {"label": "pentagon", "value": "pentagon"},
    {"label": "pentagon-open", "value": "pentagon-open"},
    {"label": "pentagon-dot", "value": "pentagon-dot"},
    {"label": "pentagon-open-dot", "value": "pentagon-open-dot"},
    {"label": "hexagon", "value": "hexagon"},
    {"label": "hexagon-open", "value": "hexagon-open"},
    {"label": "hexagon-dot", "value": "hexagon-dot"},
    {"label": "hexagon-open-dot", "value": "hexagon-open-dot"},
    {"label": "hexagon2", "value": "hexagon2"},
    {"label": "hexagon2-open", "value": "hexagon2-open"},
    {"label": "hexagon2-dot", "value": "hexagon2-dot"},
    {"label": "hexagon2-open-dot", "value": "hexagon2-open-dot"},
    {"label": "octagon", "value": "octagon"},
    {"label": "octagon-open", "value": "octagon-open"},
    {"label": "octagon-dot", "value": "octagon-dot"},
    {"label": "octagon-open-dot", "value": "octagon-open-dot"},
    {"label": "star", "value": "star"},
    {"label": "star-open", "value": "star-open"},
    {"label": "star-dot", "value": "star-dot"},
    {"label": "star-open-dot", "value": "star-open-dot"},
    {"label": "hexagram", "value": "hexagram"},
    {"label": "hexagram-open", "value": "hexagram-open"},
    {"label": "hexagram-dot", "value": "hexagram-dot"},
    {"label": "hexagram-open-dot", "value": "hexagram-open-dot"},
    {"label": "star-triangle-up", "value": "star-triangle-up"},
    {"label": "star-triangle-up-open", "value": "star-triangle-up-open"},
    {"label": "star-triangle-up-dot", "value": "star-triangle-up-dot"},
    {
        "label": "star-triangle-up-open-dot",
        "value": "star-triangle-up-open-dot",
    },
    {"label": "star-triangle-down", "value": "star-triangle-down"},
    {
        "label": "star-triangle-down-open",
        "value": "star-triangle-down-open",
    },
    {"label": "star-triangle-down-dot", "value": "star-triangle-down-dot"},
    {
        "label": "star-triangle-down-open-dot",
        "value": "star-triangle-down-open-dot",
    },
    {"label": "star-square", "value": "star-square"},
    {"label": "star-square-open", "value": "star-square-open"},
    {"label": "star-square-dot", "value": "star-square-dot"},
    {"label": "star-square-open-dot", "value": "star-square-open-dot"},
    {"label": "star-diamond", "value": "star-diamond"},
    {"label": "star-diamond-open", "value": "star-diamond-open"},
    {"label": "star-diamond-dot", "value": "star-diamond-dot"},
    {"label": "star-diamond-open-dot", "value": "star-diamond-open-dot"},
    {"label": "diamond-tall", "value": "diamond-tall"},
    {"label": "diamond-tall-open", "value": "diamond-tall-open"},
    {"label": "diamond-tall-dot", "value": "diamond-tall-dot"},
    {"label": "diamond-tall-open-dot", "value": "diamond-tall-open-dot"},
    {"label": "diamond-wide", "value": "diamond-wide"},
    {"label": "diamond-wide-open", "value": "diamond-wide-open"},
    {"label": "diamond-wide-dot", "value": "diamond-wide-dot"},
    {"label": "diamond-wide-open-dot", "value": "diamond-wide-open-dot"},
    {"label": "hourglass", "value": "hourglass"},
    {"label": "hourglass-open", "value": "hourglass-open"},
    {"label": "bowtie", "value": "bowtie"},
    {"label": "bowtie-open", "value": "bowtie-open"},
    {"label": "circle-cross", "value": "circle-cross"},
    {"label": "circle-cross-open", "value": "circle-cross-open"},
    {"label": "circle-x", "value": "circle-x"},
    {"label": "circle-x-open", "value": "circle-x-open"},
    {"label": "square-cross", "value": "square-cross"},
    {"label": "square-cross-open", "value": "square-cross-open"},
    {"label": "square-x", "value": "square-x"},
    {"label": "square-x-open", "value": "square-x-open"},
    {"label": "diamond-cross", "value": "diamond-cross"},
    {"label": "diamond-cross-open", "value": "diamond-cross-open"},
    {"label": "diamond-x", "value": "diamond-x"},
    {"label": "diamond-x-open", "value": "diamond-x-open"},
    {"label": "cross-thin", "value": "cross-thin"},
    {"label": "cross-thin-open", "value": "cross-thin-open"},
    {"label": "x-thin", "value": "x-thin"},
    {"label": "x-thin-open", "value": "x-thin-open"},
    {"label": "asterisk", "value": "asterisk"},
    {"label": "asterisk-open", "value": "asterisk-open"},
    {"label": "hash", "value": "hash"},
    {"label": "hash-open", "value": "hash-open"},
    {"label": "hash-dot", "value": "hash-dot"},
    {"label": "hash-open-dot", "value": "hash-open-dot"},
    {"label": "y-up", "value": "y-up"},
    {"label": "y-up-open", "value": "y-up-open"},
    {"label": "y-down", "value": "y-down"},
    {"label": "y-down-open", "value": "y-down-open"},
    {"label": "y-left", "value": "y-left"},
    {"label": "y-left-open", "value": "y-left-open"},
    {"label": "y-right", "value": "y-right"},
    {"label": "y-right-open", "value": "y-right-open"},
    {"label": "line-ew", "value": "line-ew"},
    {"label": "line-ew-open", "value": "line-ew-open"},
    {"label": "line-ns", "value": "line-ns"},
    {"label": "line-ns-open", "value": "line-ns-open"},
    {"label": "line-ne", "value": "line-ne"},
    {"label": "line-ne-open", "value": "line-ne-open"},
    {"label": "line-nw", "value": "line-nw"},
    {"label": "line-nw-open", "value": "line-nw-open"},
    {"label": "arrow-up", "value": "arrow-up"},
    {"label": "arrow-up-open", "value": "arrow-up-open"},
    {"label": "arrow-down", "value": "arrow-down"},
    {"label": "arrow-down-open", "value": "arrow-down-open"},
    {"label": "arrow-left", "value": "arrow-left"},
    {"label": "arrow-left-open", "value": "arrow-left-open"},
    {"label": "arrow-right", "value": "arrow-right"},
    {"label": "arrow-right-open", "value": "arrow-right-open"},
    {"label": "arrow-bar-up", "value": "arrow-bar-up"},
    {"label": "arrow-bar-up-open", "value": "arrow-bar-up-open"},
    {"label": "arrow-bar-down", "value": "arrow-bar-down"},
    {"label": "arrow-bar-down-open", "value": "arrow-bar-down-open"},
    {"label": "arrow-bar-left", "value": "arrow-bar-left"},
    {"label": "arrow-bar-left-open", "value": "arrow-bar-left-open"},
    {"label": "arrow-bar-right", "value": "arrow-bar-right"},
    {"label": "arrow-bar-right-open", "value": "arrow-bar-right-open"},
)