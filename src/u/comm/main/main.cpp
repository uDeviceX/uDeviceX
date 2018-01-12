#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

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
        j = frag_anti(i);
        cs = sb->counts[i];
        cr = rb->counts[j];
        
        if (cs != cr) ERR("%d != %d\n", cs, cr);
        comp((const int*) sb->data[i], (const int*) rb->data[j], cs);
    }
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);

    hBags sendB, recvB;
    Comm *comm;
    int capacity[NBAGS];
    float maxdensity = 26.f;
    frag_estimates(NBAGS, maxdensity, /**/ capacity);

    UC(bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &sendB, NULL));
    UC(bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &recvB, NULL));
    UC(comm_ini(m::cart, /**/ &comm));

    fill_bags(&sendB);

    UC(post_recv(&recvB, comm));
    UC(post_send(&sendB, comm));

    UC(wait_recv(comm, &recvB));
    UC(wait_send(comm));

    compare(&sendB, &recvB);

    msg_print("Passed");
    
    UC(bags_fin(HST_ONLY, NONE, &sendB, NULL));
    UC(bags_fin(HST_ONLY, NONE, &recvB, NULL));
    UC(comm_fin(/**/ comm));
    
    m::fin();
}
