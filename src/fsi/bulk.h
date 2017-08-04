namespace fsi {
void bulk(std::vector<ParticlesWrap> wsolutes) {
    if (wsolutes.size() == 0) return;

    fsi::setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart,
               wsolvent->cellscount);



    for (std::vector<ParticlesWrap>::iterator it = wsolutes.begin();
         it != wsolutes.end(); ++it)
    if (it->n)
    k_fsi::bulk<<<k_cnf(3 * it->n)>>>
        ((float2 *)it->p, it->n, wsolvent->n, local_trunk->get_float(), (float *)it->f, (float *)wsolvent->f);

}
}
