namespace wall {
struct Wa { /* local wall data */
    sdf::Tex_t texsdf;
    Texo<int> texstart;
    Texo<float4> texpp;
    int w_n;
};

namespace grey {
void force(sdf::Tex_t texsdf, hforces::Cloud cloud, const int n, const Texo<int> texstart,
           const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Wa wa, Force *ff);
}

namespace color {
void force(sdf::Tex_t texsdf, hforces::Cloud cloud, const int n, const Texo<int> texstart,
           const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Wa wa, Force *ff);
}
} /* namespace */
