#include <stdlib.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/dev.h"

#include "d/api.h"
#include "utils/cc.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "type.h"

void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m) {
    CC(d::Malloc((void**) &m->counts,  nfrags      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (nfrags + 1) * sizeof(int)));

    CC(d::alloc_pinned((void**) &m->hcounts, nfrags * sizeof(int)));
    
    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = capacity[i];
        if (c) CC(d::Malloc((void**) &m->ids[i], c * sizeof(int)));
        else   m->ids[i] = NULL;
    }
    
}

void dmap_fin(int nfrags, /**/ DMap *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));
    CC(d::FreeHost(m->hcounts));    
    for (int i = 0; i < nfrags; ++i)
        if (m->ids[i])
            CC(d::Free(m->ids[i]));
}

void dmap_reini(int nfrags, /**/ DMap m) {
    CC(d::MemsetAsync(m.counts, 0, nfrags * sizeof(int)));
}

void dmap_download_counts(int nfrags, /**/ DMap *m) {
    CC(d::MemcpyAsync(m->hcounts, m->counts, nfrags * sizeof(int), D2H));
}

void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m) {
    UC(emalloc( nfrags      * sizeof(int), (void**) &m->counts));
    UC(emalloc((nfrags + 1) * sizeof(int), (void**) &m->starts));

    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = capacity[i];
        if (c)
            UC(emalloc(c * sizeof(int), (void**) &m->ids[i]));
    }
}

void dmap_fin_host(int nfrags, /**/ DMap *m) {
    free(m->counts);
    free(m->starts);    
    for (int i = 0; i < nfrags; ++i)
        free(m->ids[i]);
}

void dmap_reini_host(int nfrags, /**/ DMap m) {
    memset(m.counts, 0, nfrags * sizeof(int));
}

void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h) {
    CC(d::MemcpyAsync(h->counts, d->counts,  nfrags      * sizeof(int), D2H));
    CC(d::MemcpyAsync(h->starts, d->starts, (nfrags + 1) * sizeof(int), D2H));

    dSync();
    
    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = h->counts[i];
        if (c)
            CC(d::MemcpyAsync(h->ids[i], d->ids[i], c * sizeof(int), D2H));
    }
}
