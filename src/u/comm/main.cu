#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mpi/basetags.h"
#include "glb.h"
#include "frag.h"

#include "comm/imp.h"

/* generate a unique sequence given a unique val */
void fill_bag(int val, int sz, int *ii) {
    for (int i = 0; i < sz; ++i) ii[i] = -2*val + val*val;
}

void fill_bags(comm::hBags *b) {
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

void compare(const comm::hBags *sb, const comm::hBags *rb) {
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
    comm::hBags sendB, recvB;
    comm::dBags senddB, recvdB;
    comm::Stamp stamp;

    ini(/**/ &tg);
    ini_no_bulk(sizeof(int), 26, /**/ &sendB, &senddB);
    ini_no_bulk(sizeof(int), 26, /**/ &recvB, &recvdB);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);

    fill_bags(&sendB);

    post_recv(&recvB, &stamp);
    post_send(&sendB, &stamp);

    wait_recv(&stamp, &recvB);
    wait_send(&stamp);

    compare(&sendB, &recvB);

    MSG("Passed");
    
    fin(&sendB, &senddB);
    fin(&recvB, &recvdB);
    fin(/**/ &stamp);
    
    m::fin();
}
