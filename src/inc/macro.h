/* texture size for wall */
enum {
    XSIZE_WALLCELLS = 2 * XWM + XS,
    YSIZE_WALLCELLS = 2 * YWM + YS,
    ZSIZE_WALLCELLS = 2 * ZWM + ZS,
    XE = XS + 2*XWM, YE = YS + 2*YWM, ZE = ZS + 2*ZWM,
    XTE  = 16*16,/* texture sizes */
    _YTE = ceiln(YE*XTE, XE), YTE  = 16*ceiln(_YTE, 16),
    _ZTE = ceiln(ZE*XTE, XE), ZTE  = 16*ceiln(_ZTE, 16)
};
