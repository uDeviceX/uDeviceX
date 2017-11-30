static __device__ bool is_dead(int i, const PartList lp) {
    if (lp.deathlist)
        return lp.deathlist[i] == DEAD;
    return false;
}

static __device__ bool is_alive(int i, const PartList lp) {
    return !is_dead(i, lp);
}
