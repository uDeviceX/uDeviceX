#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "conf/imp.h"
#include "coords/ini.h"
#include "coords/imp.h"
#include "utils/error.h"

#include "frag/imp.h"

#include "comm/imp.h"

/* generate a unique sequence given a unique val */
void fill_bag(int val, int sz, int *ii) {
    for (int i = 0; i < sz; ++i) ii[i] = -2*val + val*val;
}

void fill_bags(hBags *b) {
    int c, i;
    for (i = 0; i < 26; ++i) {
        c = i;
        fill_bag(i, c, (int*) b->data[i]);
        b->counts[i] = c;
    }
}

void comp(const int *aa, const int *bb, int n) {
    int i, a, b;
    for (i = 0; i < n; ++i) {
        a = aa[i];
        b = bb[i];
        if (a != b)
            ERR("%d != %d for i = %d\n", a, b, i);
    }
}

void compare(const hBags *sb, const hBags *rb) {
    int i, j, cs, cr;
    for (i = 0; i < 26; ++i) {
        j = frag_hst::anti(i);
        cs = sb->counts[i];
        cr = rb->counts[j];
        
        if (cs != cr) ERR("%d != %d\n", cs, cr);
        comp((const int*) sb->data[i], (const int*) rb->data[j], cs);
    }
}

int main(int argc, char **argv) {
    enum {N = 3};
    hBags sendB[N], recvB[N];
    Comm *comm;
    CommBuffer *scommbuf, *rcommbuf;
    int capacity[NBAGS];
    float maxdensity = 26.f;
    int3 L;
    Config *cfg;
    Coords *coords;
    int i, rank, size, dims[3];
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);
    
    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(coords_ini_conf(cart, cfg, &coords));

    L = subdomain(coords);

    frag_hst::estimates(L, NBAGS, maxdensity, /**/ capacity);

    for (i = 0; i < N; ++i) {
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &sendB[i], NULL));
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &recvB[i], NULL));
        fill_bags(&sendB[i]);
    }
    
    UC(comm_ini(cart, /**/ &comm));
    UC(comm_buffer_ini(N, sendB, &scommbuf));
    UC(comm_buffer_ini(N, recvB, &rcommbuf));


    UC(comm_buffer_set(N, sendB, scommbuf));

    UC(comm_post_recv(rcommbuf, comm));
    UC(comm_post_send(scommbuf, comm));

    UC(comm_wait_recv(comm, rcommbuf));
    UC(comm_wait_send(comm));

    UC(comm_buffer_get(rcommbuf, N, recvB));

    for (i = 0; i < N; ++i)
        compare(&sendB[i], &recvB[i]);

    msg_print("Passed");

    for (i = 0; i < N; ++i) {
        UC(comm_bags_fin(HST_ONLY, NONE, &sendB[i], NULL));
        UC(comm_bags_fin(HST_ONLY, NONE, &recvB[i], NULL));
    }
    UC(comm_fin(/**/ comm));
    UC(comm_buffer_fin(scommbuf));
    UC(comm_buffer_fin(rcommbuf));
    
    UC(coords_fin(coords));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}
