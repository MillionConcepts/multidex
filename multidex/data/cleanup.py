"""
Clean-ups applied to data on load.  This is intentionally minimal -
just things that would be a pain to do on the front end, and/or
that unnecessarily bulk up the JSON blobs sent between back and front.
"""

from ast import literal_eval

import pyarrow as pa


def cleanup(tbl: pa.Table, *, have_browse_images: bool) -> pa.Table:
    return decode_image_json(
        drop_all_null_columns(tbl),
        have_browse_images = have_browse_images,
    )


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


def decode_image_json(tbl: pa.Table, *, have_browse_images: bool) -> pa.Table:
    """
    Return (a copy of) 'tbl' with the "images" column, which contains
    stringified JSON structures like

        {"left": "filename", "right": "filename"}

    (except sometimes with single-quoted strings, which is why we use
    literal_eval instead of json.loads below) replaced with a set of
    "image_xxx" columns that directly hold the file names.

    If there isn't an "images" column, 'tbl' is returned unmodified.
    If there are both "images" and "image_*" columns, we issue a warning
    and return (a copy of) 'tbl' with the "images" column dropped and
    the "image_*" columns unaltered (this may change in the future).
    If "images" column cells can't be parsed, we issue a warning and
    return 'tbl' unmodified.

    If 'have_browse_images' is false, instead we return (a copy of) 'tbl'
    with "images" and all "image_*" columns dropped, and no other changes.
    In this case we do not check for any consistency issues among the
    image columns.
    """
    if not have_browse_images:
        return tbl.drop_columns([
            col for col in tbl.column_names
            if col == "images" or col.startswith("image_")
        ])

    if "images" not in tbl.column_names:
        return tbl

    if any(col.startswith("image_") for col in tbl.column_names):
        import warnings
        warnings.warn("table has both images and image_* columns, "
                      "leaving images column alone")
        return tbl

    blobs = []
    columns = {}
    for cell in tbl["images"]:
        try:
            blob = literal_eval(cell.as_py())
            blobs.append(blob)
            for eye in blob.keys():
                if eye not in columns:
                    columns[eye] = []

        except Exception as e:
            import warnings
            warnings.warn(f"parse error in images cell {cell!r}: {e}; "
                          "leaving images column alone")
            return tbl

    for blob in blobs:
        for eye, col in columns.items():
            col.append(blob.get(eye))

    tbl = tbl.drop_columns(["images"])
    for eye, col in columns.items():
        tbl = tbl.append_column("image_"+eye, pa.array(col))
    return tbl
