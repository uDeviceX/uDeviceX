void force(const sdf::Quants *qsdf, const Quants q, const Ticket t, Cloud cloud, const int n, Force *ff) {
    Wa wa; /* local wall data */
    wa.sdf = qsdf->texsdf;
    wa.start = t.texstart;
    wa.pp  = t.texpp;
    wa.n      = q.n;

    force(cloud, n, t.rnd, wa, /**/ ff);
}
