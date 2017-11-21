namespace hforces {

/* fragment */
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

struct Rnd {
    float seed;
    int mask;
};

typedef Sarray<int, 27> int27;
typedef Sarray<LFrag, 26> LFrag26;
typedef Sarray<RFrag, 26> RFrag26;
typedef Sarray<  Rnd, 26>   Rnd26;

void interactions(const LFrag26 lfrags, const RFrag26 rfrags, const Rnd26 rrnd, /**/ float *ff);

} /* namespace */
