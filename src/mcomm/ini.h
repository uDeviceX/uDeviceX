namespace mcomm {
namespace sub {

void ini_tcom(MPI_Comm cart, /**/ MPI_Comm *newcart, int rnk_ne[27], int ank_ne[27]) {
    int crds[3];

    rnk_ne[0] = ank_ne[0] = m::rank;
    for (int i = 1; i < 27; ++i) {
        const int d[3] = {(i     + 1) % 3 - 1,
                          (i / 3 + 1) % 3 - 1,
                          (i / 9 + 1) % 3 - 1};

        for (int c = 0; c < 3; ++c) crds[c] = m::coords[c] + d[c];
        MC(l::m::Cart_rank(cart, crds, rnk_ne + i));
        for (int c = 0; c < 3; ++c) crds[c] = m::coords[c] - d[c];
        MC(l::m::Cart_rank(cart, crds, ank_ne + i));
    }
    MC(l::m::Comm_dup(cart, /**/ newcart));
}

} // sub
} // mcomm
