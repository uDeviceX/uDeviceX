namespace fsi {
void bulk(std::vector<ParticlesWrap> wr) {
    if (wr.size() == 0) return;
    setup(wo->p, wo->n, wo->cellsstart);
    for (std::vector<ParticlesWrap>::iterator it = wr.begin(); it != wr.end(); ++it)
        KL(k_fsi::bulk,
           (k_cnf(3*it->n)),
           ((float2 *)it->p, it->n, wo->n, rgen->get_float(), (float*)it->f, (float*)wo->f));
}
}
