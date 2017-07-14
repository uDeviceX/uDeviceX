/* fragment */
enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct Frag {
    float *xdst;
    int *ii;

    float2 *xsrc;
    int ndst, nsrc, *cellstarts, dx, dy, dz, xcells, ycells, zcells;
    FragType type;
};

struct Rnd {
    float seed;
    int mask;
};
