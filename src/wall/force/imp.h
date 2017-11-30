namespace wall {
struct Wa { /* local wall data */
    sdf::tex3Dca sdf;
    Texo<int> start;
    Texo<float4> pp;
    int n;
};

namespace grey {
void force(Cloud cloud, int n, rnd::KISS *rnd, Wa wa, /**/ Force *ff);
}

namespace color {
void force(Cloud cloud, int n, rnd::KISS *rnd, Wa wa, /**/ Force *ff);
}

} /* namespace */
