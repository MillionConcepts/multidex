"""
Clean-ups applied to data on load.  This is intentionally minimal -
just things that would be a pain to do on the front end, and/or
that unnecessarily bulk up the JSON blobs sent between back and front.
"""

from ast import literal_eval

import pyarrow as pa


def cleanup(tbl: pa.Table) -> pa.Table:
    return decode_image_json(drop_all_null_columns(tbl))


def drop_all_null_columns(tbl: pa.Table) -> pa.Table:
    """
    Return (a copy of) 'tbl' with columns containing only nulls
    removed.
    """
    null_columns = []
    # ??? I think this is the most efficient way to do this.
    # Note that Table.drop_null drops *rows* that contain *any* null
    # values, which is not what we want.
    for name, col in zip(tbl.column_names, tbl.columns):
        if col.null_count == col.length():
            null_columns.append(name)
    return tbl.drop_columns(null_columns)


def decode_image_json(tbl: pa.Table) -> pa.Table:
    """
    Return (a copy of) 'tbl' with the "images" column, which contains
    stringified JSON structures like '{"left": "filename", "right": "filename"}'
    (except sometimes with single-quoted strings, which is why we use
    literal_eval instead of json.loads below) replaced with "image_left"
    and "image_right" columns that directly hold the file names.

    If there isn't an "images" column, 'tbl' is returned unmodified.
    If there is an "images" column but it's not as we expect, a warning
    message is generated and 'tbl' is returned unmodified.
    """
    if "images" not in tbl.column_names:
        return tbl

    images_left = []
    images_right = []
    for cell in tbl["images"]:
        try:
            blob = literal_eval(cell.as_py())
            left = blob.pop("left", None)
            right = blob.pop("right", None)
            if blob:
                raise ValueError("unexpected keys: " + ", ".join(blob.keys()))
            images_left.append(left)
            images_right.append(right)
        except Exception as e:
            import warnings
            warnings.warn(f"images cell {cell!r} not as expected: {e}")
            warnings.warn("leaving images column alone")
            return tbl

    return (
        tbl.drop_columns("images")
           .append_column("images_left", pa.array(images_left))
           .append_column("images_right", pa.array(images_right))
    )
