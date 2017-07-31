namespace bibsbatch {

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

void interactions(const SFrag ssfrag[26], const Frag ffrag[26], const Rnd rrnd[26], /**/ float *ff);
};
