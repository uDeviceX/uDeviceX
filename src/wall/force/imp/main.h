void force(sdf::Tex_t texsdf, hforces::Cloud cloud, const int n, const Texo<int> texstart,
           const Texo<float4> texpp, const int w_n, rnd::KISS *rnd, Wa wa, /**/ Force *ff) {
    KL(dev::force,
       (k_cnf(3*n)),
       (texsdf, cloud, n, w_n, rnd->get_float(), texstart, texpp, /**/ (float*)ff));
}
