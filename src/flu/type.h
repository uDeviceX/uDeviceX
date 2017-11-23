/* helpers to communicate between halo interactions and exchanger */

namespace flu {

enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag { /* "local" fragment */
    Cloud c;
    const int *ii; /* index */
    int n;
};

struct RFrag { /* "remote" fragment */
    Cloud c;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    FragType type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

} // flu
