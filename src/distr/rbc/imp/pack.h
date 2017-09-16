template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, NBAGS> *buf) {
    for (int i = 0; i < NBAGS; ++i)
        buf->d[i] = (T*) bags.data[i];
}

static void pack_pp(const Map m, int nc, int nv, const Particle *pp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);
        
    KL((dev::pack_pp), (blck, thrd), (nv, pp, m, /**/ wrap));
}

