static void frag_set_buffer(int fid, int nbags, const hBags *hbb, CommBuffer *cb) {
    size_t tot_sz, sz, bs;
    int i, c, *counts;
    data_t *d;
    const hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    tot_sz = sz = nbags * sizeof(int);
    d += sz;

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        bs = hb->bsize;
        c  = hbb->counts[fid];
        sz = c * bs;
        memcpy(d, hbb->data[fid], sz);
        counts[i] = c;
        d += sz;
        tot_sz += sz;
    }
    cb->sz[fid] = tot_sz;

    if (tot_sz > cb->cap[fid]) ERR("fragment %d: exceed buffer size %ld/%ld\n", fid, tot_sz, cb->cap[fid]);
}

void comm_buffer_set(int nbags, const hBags *hbb, CommBuffer *cb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_set_buffer(fid, nbags, hbb, cb);
}

static void frag_get_buffer(int fid, const CommBuffer *cb, int nbags, hBags *hbb) {
    size_t sz;
    int i, c, *counts;
    data_t *d;
    hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    d += nbags * sizeof(int);
    
    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        c = counts[i];
        hb->counts[fid] = c;
        sz = c * hb->bsize;
        memcpy(hb->data, d, sz);
        d += sz;
    }
}

void comm_buffer_get(const CommBuffer *cb, int nbags, hBags *hbb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_get_buffer(fid, cb, nbags, hbb);    
}
