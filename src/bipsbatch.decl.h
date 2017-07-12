namespace bipsbatch {

/* fragment */
enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct BatchInfo {
    float *xdst;
    float2 *xsrc;
    float seed;
    int ndst, nsrc, mask, *cellstarts, *scattered_entries, dx, dy, dz, xcells, ycells, zcells;
    FragType halotype;
};

struct Rnd {
    float seed;
    int mask;
};
}
