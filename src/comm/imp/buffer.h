static void resize(int i, size_t sz, CommBuffer *cb) {
    enum {
        STRIDE = 128
    };
    size_t cap;
    if (cb->cap[i] < sz) {
        if (cb->buf[i]) UC(efree(cb->buf[i]));
        cap = ceiln(sz, STRIDE) * STRIDE; 
        UC(emalloc(cap, (void**) &cb->buf[i]));
        cb->cap[i] = cap;
    }
    cb->sz[i] = sz;
}

static void frag_set_buffer(int fid, int nbags, const hBags *hbb, CommBuffer *cb) {
    size_t sz, bs;
    int i, c, *counts;
    data_t *d;
    const hBags *hb;
    sz = nbags * sizeof(int);

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        bs = hb->bsize;
        c  = hbb->counts[fid];        
        sz += c * bs;
    }

    resize(fid, sz, cb);

    d = cb->buf[fid];
    counts = (int*) d;
    d += nbags * sizeof(int);

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        bs = hb->bsize;
        c  = hbb->counts[fid];
        sz = c * bs;
        memcpy(d, hbb->data[fid], sz);
        counts[i] = c;
        d += sz;
    }
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
