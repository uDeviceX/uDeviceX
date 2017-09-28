namespace wall {
typedef const sdf::tex3Dca<float> TexSDF_t;
struct Wa { /* local wall data */
    TexSDF_t texsdf;
    Texo<float4> texpp;
    Texo<int> texstart;
    int w_n;
};

namespace grey {
void force(TexSDF_t texsdf, hforces::Cloud cloud, const int n, const Texo<int> texstart,
           const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Force *ff);
}

namespace color {
void force(TexSDF_t texsdf, hforces::Cloud cloud, const int n, const Texo<int> texstart,
           const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Force *ff);
}
} /* namespace */
