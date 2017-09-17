#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mpi/basetags.h"
#include "glb.h"
#include "frag/imp.h"

#include "comm/imp.h"

using namespace comm;

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
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Comm unit test!");

    basetags::TagGen tg;
    hBags sendB, recvB;
    Stamp stamp;

    ini(/**/ &tg);
    ini(HST_ONLY, NONE, sizeof(int), 26, /**/ &sendB, NULL);
    ini(HST_ONLY, NONE, sizeof(int), 26, /**/ &recvB, NULL);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);

    fill_bags(&sendB);

    post_recv(&recvB, &stamp);
    post_send(&sendB, &stamp);

    wait_recv(&stamp, &recvB);
    wait_send(&stamp);

    compare(&sendB, &recvB);

    MSG("Passed");
    
    fin(HST_ONLY, NONE, &sendB, NULL);
    fin(HST_ONLY, NONE, &recvB, NULL);
    fin(/**/ &stamp);
    
    m::fin();
}
