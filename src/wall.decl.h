enum {
    XSIZE_WALLCELLS = 2 * XWM + XS,
    YSIZE_WALLCELLS = 2 * YWM + YS,
    ZSIZE_WALLCELLS = 2 * ZWM + ZS,

    XTE = 256,
    _YTEXTURESIZE = ((YS + 2 * YWM) * XTE +
                     XS + 2 * XWM - 1) /
    (XS + 2 * XWM),

    YTE = 16 * ((_YTEXTURESIZE + 15) / 16),
    _ZTEXTURESIZE = ((ZS + 2 * ZWM) * XTE +
                     XS + 2 * XWM - 1) /
    (XS + 2 * XWM),
    ZTE = 16 * ((_ZTEXTURESIZE + 15) / 16),
};

namespace wall {
Logistic::KISS* trunk;

int w_n;
float4 *w_pp;
x::Clist *wall_cells;
}
