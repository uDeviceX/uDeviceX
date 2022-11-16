/* helpers to communicate between halo interactions and exchanger */

namespace flu {

enum { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag { /* "local" fragment */
    PaArray parray;
    const int *ii; /* index */
    int n;
};

struct RFrag { /* "remote" fragment */
    PaArray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

} // flu
