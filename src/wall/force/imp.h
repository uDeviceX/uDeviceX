namespace wall {
struct Wa { /* local wall data */
    sdf::Tex_t texsdf;
    Texo<int> texstart;
    Texo<float4> texpp;
    int w_n;
};

namespace grey {
void force(hforces::Cloud cloud, const int n, rnd::KISS *rnd, Wa wa, /**/ Force *ff);
}

namespace color {
void force(hforces::Cloud cloud, const int n, rnd::KISS *rnd, Wa wa, /**/ Force *ff);
           
}
} /* namespace */
