void pair(TexSDF_t texsdf, const int type, hforces::Cloud cloud, const int n, const Texo<int> texstart,
                  const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Force *ff) {
    KL(dev::pair,
       (k_cnf(3*n)),
       (texsdf, cloud, n, w_n, (float *)ff, rnd->get_float(), type, texstart, texpp));
}
