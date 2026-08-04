"""hier_hmm — a two-level phase-type HMM for single-molecule methylation data.

Level 1 segments each molecule into open / linker / nucleosome using duration
priors (a nucleosome is ~129 bp with a hard 60 bp floor, not geometrically
distributed) and a grammar. Level 2 searches the interior of the open runs at
base-pair resolution for short protein footprints. The emission is a masked
binomial on (methylated, callable) counts, so a bin's influence scales with how
many callable positions it actually has.

Every prior, threshold and bin size lives in `default_config.yaml`; the code
holds no second copy of them.

    from hier_hmm import load_config, prepare, run
    cfg = load_config("my.yaml")
    ds  = prepare(m6a_array, cfg)          # NaN = not callable, 0/1 = un/methylated
    res = run(ds, cfg)
    res.labels                              # (n_fiber, n_bp) int8, -1 = no data
"""
__version__ = "0.1.0"

from .config import default_config, dump_config, load_config, state_labels  # noqa: F401
from .data import Dataset, Fiber, load_npz, prepare                          # noqa: F401
from .model import NODATA, HierHMM                                           # noqa: F401
from .runner import Result, run, save                                        # noqa: F401

__all__ = ["load_config", "default_config", "dump_config", "state_labels",
           "prepare", "load_npz", "Dataset", "Fiber",
           "HierHMM", "NODATA", "run", "save", "Result", "__version__"]
