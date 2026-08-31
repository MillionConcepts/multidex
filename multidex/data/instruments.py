"""
Data processing operations that are intrinsically dependent on the
instrument that collected the data.
"""

# Note: there should be no imports from marslab at file scope,
# because importing marslab currently takes a long time:
# <https://github.com/MillionConcepts/marslab/issues/31>
# Delaying the imports to function scope means they can run
# somewhat in parallel with front-end initialization, and
# makes --help complete promptly.

from dataclasses import dataclass
from typing import Sequence, TypedDict, cast

import pyarrow as pa
import pyarrow.compute as pc

# A close-enough approximation to the actual shape of DERIVED_CAM_DICT entries.
CamInfo = TypedDict("CamInfo", {
    "filters": dict[str, int],
    "virtual_filters": dict[str, int],
    "virtual_filter_mapping": dict[str, Sequence[str]],
    "canonical_averaged_filters": dict[str, int],
})


@dataclass(frozen=True, slots=True)
class Instrument:
    # Name of the instrument.
    name: str

    # This instrument's entry in DERIVED_CAM_DICT
    cam_info: CamInfo

    # Number of (electric) eyes on the instrument.  Currently this
    # is always either 1 or 2, but more is possible in the future.
    n_eyes: int

    def compute_spectrum(
        self,
        rowid: int,
        dataset: pa.Table,
        *,
        scale: bool | tuple[str, str] = True,
        average_filters: bool = True,
        show_bayers: bool = True,
    ) -> dict[str, object]:
        from marslab.compat.xcam import polish_xcam_spectrum

        # TODO: Handle monocular instruments (may need changes in marslab)
        if self.n_eyes != 2:
            eyes = "eye" if self.n_eyes == 1 else "eyes"
            raise RuntimeError(
                f"sorry, not implemented: spectra for {self.n_eyes} {eyes}"
            )

        # runtime arg validation is intentionally a little more lenient than
        # the type signature
        if scale in (None, False):
            scale_to = None
        elif scale is True:
            # FIXME: this only works for M/Z/P cams, not C or S.
            # C and S are monocular, M/Z/P binocular, so this is closely
            # related to the TODO above, but there's no intrinsic reason
            # why binocular cams _have_ to have filters named "L1" and "R1".
            scale_to = ("L1", "R1")
        elif (isinstance(scale, tuple) and len(scale) == 2):
            # Can only scale to a pair of physical filters
            if not (scale[0] in self.cam_info["filters"]
                    and scale[1] in self.cam_info["filters"]):
                raise ValueError(f"cannot scale to ({scale[0]},{scale[1]}):"
                                 f" some filters are unavailable")
            scale_to = scale
        else:
            raise TypeError("'scale' argument must be a boolean"
                            " or a pair of filters")

        record = dataset.filter(pc.field("id") == rowid)
        if record.num_rows == 0:
            raise ValueError(f"no data for observation {rowid}")
        if record.num_rows > 1:
            raise RuntimeError(f"observation {rowid} is not unique")

        avail = { n.lower(): n
                  for n in record.column_names }
        spectrum = {}
        drop_bayers = not show_bayers

        for filt in self.cam_info["filters"].keys():
            if drop_bayers and filt[-1] in "RrGgBb":
                continue

            f_std = filt + "_STD"
            m_col = avail.get(filt.lower())
            s_col = avail.get(f_std.lower())
            if m_col is None or s_col is None:
                continue
            spectrum[filt] = record.column(m_col)[0].as_py()
            spectrum[f_std] = record.column(s_col)[0].as_py()

        return cast(dict[str, object], polish_xcam_spectrum(
            spectrum = spectrum,
            cam_info = self.cam_info,
            scale_to = scale_to,
            average_filters = average_filters,
        ))


ALL_PHYS_FILTERS = None
def all_phys_filters() -> frozenset[str]:
    global ALL_PHYS_FILTERS
    if ALL_PHYS_FILTERS is None:
        from marslab.compat.xcam import DERIVED_CAM_DICT

        pfs: set[str] = set()
        for cam_info in DERIVED_CAM_DICT.values():
            pfs.update(n.lower() for n in cam_info["filters"].keys())

        ALL_PHYS_FILTERS = frozenset(pfs)
    return ALL_PHYS_FILTERS


def identify_instrument(tbl: pa.Table) -> Instrument:
    """
    Given a table of data collected by one of the xCAM instruments,
    figure out which of them it was.

    TODO: This should probably be explicit table metadata rather than
    being worked out by heuristic analysis of the table's columns.
    """
    from marslab.compat.xcam import DERIVED_CAM_DICT, WAVELENGTH_TO_FILTER

    all_columns = frozenset(n.lower() for n in tbl.column_names)
    phys_filter_columns = all_columns & all_phys_filters()

    def unknown_instrument() -> Exception:
        return ValueError(
            "unable to identify instrument for this dataset; columns are: "
            + ", ".join(sorted(all_columns))
        )

    candidates = []
    for name, cam_info in DERIVED_CAM_DICT.items():
        if phys_filter_columns == set(
            n.lower() for n in cam_info["filters"].keys()
        ):
            candidates.append(name)

    match len(candidates):
        case 1:
            name = candidates[0]
        case 2:
            # currently this is the only case where this happens
            c0 = candidates[0]
            c1 = candidates[1]
            if c0 not in ("MCAM", "ZCAM") or c1 not in ("MCAM", "ZCAM"):
                raise ValueError(
                    f"ambiguous physical filter set ({c0}/{c1}): "
                    + ", ".join(sorted(phys_filter_columns))
                )

            MCAM_unique = frozenset(("group", "rock_class", "soil_class"))
            ZCAM_unique = frozenset(("target", "analysis_name", "distance"))

            unique_columns = all_columns & (MCAM_unique | ZCAM_unique)
            if unique_columns == MCAM_unique:
                name = "MCAM"
            elif unique_columns == ZCAM_unique:
                name = "ZCAM"
            else:
                raise unknown_instrument()
        case _:
            raise unknown_instrument()

    cam_info = DERIVED_CAM_DICT[name]
    if set(WAVELENGTH_TO_FILTER[name].keys()) == {"L", "R"}:
        n_eyes = 2
    else:
        n_eyes = 1

    return Instrument(name, cam_info, n_eyes)
