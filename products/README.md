IsoFLOP regression products
===========================

`isoflop_points.csv` is the 60-row historical-analysis input from branch
`ss/iter-flops`, commit `8d965cd`. Its numerical fields match the original
local Figure 4 data; config paths are repository-relative. Line endings are normalized to LF;
all recorded fields are unchanged.

`isoflop_optima_legacy.json` was calculated from these points using the
unmodified `notebooks/manuscript/plot_isoflop.py` at `32af721`. It verifies
that shared log-log fitting preserves the historical fit results. These
recorded budgets/parameter counts are not recalculated by the new model
accounting convention.
