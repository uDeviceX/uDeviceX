namespace hforces {

/* fragment */
enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct SFrag { /* "send" fragment */
    float *pp;
    int *ii;
    int n;
};

struct Frag {
    float2 *pp;
    int *start, dx, dy, dz, xcells, ycells, zcells;
    FragType type;
};

struct Rnd {
    float seed;
    int mask;
};

typedef Sarray<unsigned int, 27> int27;
typedef Sarray<SFrag, 26> SFrag26;
typedef Sarray< Frag, 26>  Frag26;
typedef Sarray<  Rnd, 26>   Rnd26;

void interactions(const SFrag26 ssfrag, const Frag26 ffrag, const Rnd26 rrnd, /**/ float *ff);

} // hforces
