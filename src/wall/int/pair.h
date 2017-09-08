namespace wall {
void pair(const sdf::Quants qsdf, const Quants q, const Ticket t, const int type, hforces::Cloud cloud, const int n, Force *ff) {
    pair(qsdf.texsdf, type, cloud, n, t.texstart, t.texpp, q.n, /**/ t.rnd, ff);
}
}
