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

void fill_bag(int val, int sz, int *ii) {
    for (int i = 0; i < sz; ++i) ii[i] = val;
}

void fill_bags(comm::Bags *b) {
    int c, i;
    for (i = 0; i < 26; ++i) {
        c = i;
        fill_bag(i, c, (int*) b->hst[i]);
        b->counts[i] = c;
    }
}

bool comp(const int *a, const int *b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            printf("%d != %d for i = %d\n", a[i], b[i], i);
            return false;
        }
    return true;
}

void compare(const comm::Bags *sb, const comm::Bags *rb) {
    int i, j, cs, cr;
    for (i = 0; i < 26; ++i) {
        j = frag_anti(i);
        cs = sb->counts[i];
        cr = rb->counts[j];
        printf("%d - %d\n", cs, cr);
        assert(cs == cr);
        assert(comp((const int*) sb->hst[i], (const int*) rb->hst[j], cs));
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Comm unit test!");

    basetags::TagGen tg;
    comm::Bags sendB, recvB;
    comm::Stamp stamp;

    ini(/**/ &tg);
    ini_no_bulk(sizeof(int), 26, /**/ &sendB);
    ini_no_bulk(sizeof(int), 26, /**/ &recvB);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);

    fill_bags(&sendB);

    post_recv(&recvB, &stamp);
    post_send(&sendB, &stamp);

    wait_recv(&stamp, &recvB);
    wait_send(&stamp);

    compare(&sendB, &recvB);
    
    fin(&sendB);
    fin(&recvB);
    fin(/**/ &stamp);
    
    m::fin();
}
