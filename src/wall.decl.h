enum {
    XSIZE_WALLCELLS = 2 * XWM + XS,
    YSIZE_WALLCELLS = 2 * YMARGIN_WALL + YS,
    ZSIZE_WALLCELLS = 2 * ZMARGIN_WALL + ZS,

    XTEXTURESIZE = 256,
    _YTEXTURESIZE = ((YS + 2 * YMARGIN_WALL) * XTEXTURESIZE +
                     XS + 2 * XWM - 1) /
    (XS + 2 * XWM),

    YTEXTURESIZE = 16 * ((_YTEXTURESIZE + 15) / 16),
    _ZTEXTURESIZE = ((ZS + 2 * ZMARGIN_WALL) * XTEXTURESIZE +
                     XS + 2 * XWM - 1) /
    (XS + 2 * XWM),
    ZTEXTURESIZE = 16 * ((_ZTEXTURESIZE + 15) / 16),
};

namespace wall {
Logistic::KISS* trunk;

int w_n;
float4 *w_pp;
CellLists *wall_cells;
}
