/* fragment */
enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct SFrag { /* "send" fragment */
    float *pp;
    int *ii;
    int n;
};

struct Frag {
    float2 *pp;
    int *cellstarts, dx, dy, dz, xcells, ycells, zcells;
    FragType type;
};

struct Rnd {
    float seed;
    int mask;
};
