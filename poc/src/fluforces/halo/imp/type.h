template<typename Parray>
struct LFrag_v { /* "local" fragment */
    Parray parray;
    const int *ii; /* index */
    int n;
};

template<typename Parray>
struct RFrag_v { /* "remote" fragment */
    Parray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

template <typename Parray>
using LFrag_v26 = Sarray<LFrag_v<Parray>, 26>;

template <typename Parray>
using RFrag_v26 = Sarray<RFrag_v<Parray>, 26>;
