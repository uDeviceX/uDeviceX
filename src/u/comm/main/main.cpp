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
#include "parser/imp.h"
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

void comp(const int *a, const int *b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i])
            ERR("%d != %d for i = %d\n", a[i], b[i], i);
}

void compare(const hBags *sb, const hBags *rb) {
    int i, j, cs, cr;
    for (i = 0; i < 26; ++i) {
        j = fraghst::anti(i);
        cs = sb->counts[i];
        cr = rb->counts[j];
        
        if (cs != cr) ERR("%d != %d\n", cs, cr);
        comp((const int*) sb->data[i], (const int*) rb->data[j], cs);
    }
}

int main(int argc, char **argv) {
    hBags sendB, recvB;
    Comm *comm;
    int capacity[NBAGS];
    float maxdensity = 26.f;
    int3 L;
    Config *cfg;
    Coords *coords;
    int rank, size, dims[3];
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

    fraghst::estimates(L, NBAGS, maxdensity, /**/ capacity);

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &sendB, NULL));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &recvB, NULL));
    UC(comm_ini(cart, /**/ &comm));

    fill_bags(&sendB);

    UC(comm_post_recv(&recvB, comm));
    UC(comm_post_send(&sendB, comm));

    UC(comm_wait_recv(comm, &recvB));
    UC(comm_wait_send(comm));

    compare(&sendB, &recvB);

    msg_print("Passed");
    
    UC(comm_bags_fin(HST_ONLY, NONE, &sendB, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, &recvB, NULL));
    UC(comm_fin(/**/ comm));

    UC(coords_fin(coords));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}
