namespace cnt {
void bulk(std::vector<ParticlesWrap> wsolutes) {
    if (wsolutes.size() == 0) return;

    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];
        if (it.n)
        k_cnt::bulk_3tpp<<<k_cnf(3 * it.n)>>>
            ((float2 *)it.p, it.n, cellsentries->S, wsolutes.size(), (float *)it.f,
             local_trunk->get_float(), i);

    }
}
}
