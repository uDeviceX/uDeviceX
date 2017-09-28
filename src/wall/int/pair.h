void force(const sdf::Quants qsdf, const Quants q, const Ticket t, hforces::Cloud cloud, const int n, Force *ff) {
    force(qsdf.texsdf, cloud, n, t.texstart, t.texpp, q.n, /**/ t.rnd, ff);
}
