void force(const sdf::Quants qsdf, const Quants q, const Ticket t, hforces::Cloud cloud, const int n, Force *ff) {
    Wa wa; /* local wall data */
    wa.texsdf = qsdf.texsdf;
    wa.texstart = t.texstart;
    wa.texpp  = t.texpp;
    wa.w_n      = q.n;

    force(cloud, n, t.rnd, wa, /**/ ff);
}
