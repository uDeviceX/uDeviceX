namespace mdstr {
namespace sub {

/* decode neighbors linear index to "delta"
   0 -> { 0, 0, 0}
   1 -> { 1, 0, 0}
   ...
   20 -> {-1, 0, -1}
   ...
   26 -> {-1, -1, -1}
*/
#define i2del(i) {((i)     + 1) % 3 - 1,        \
                  ((i) / 3 + 1) % 3 - 1,        \
                  ((i) / 9 + 1) % 3 - 1}


/* generate ranks and anti-ranks of the neighbors */
void gen_ne(MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne) {
    rnk_ne[0] = ank_ne[0] = m::rank;
    for (int i = 1; i < 27; ++i) {
        int d[3] = i2del(i); /* index to delta */
        int co_ne[3];
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] + d[c];
        l::m::Cart_rank(cart, co_ne, &rnk_ne[i]);
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
        l::m::Cart_rank(cart, co_ne, &ank_ne[i]);
    }
}

} // sub
} // mdstr
