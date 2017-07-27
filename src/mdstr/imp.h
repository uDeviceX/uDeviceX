namespace mdstr {
namespace sub {

// ini
void gen_ne(MPI_Comm cart, /**/ int* rnk_ne, int* ank_ne);

// imp
void get_dests(const float *rr, int nm, /**/ int *dests[27], int counts[27]);


} // sub
} // mdstr
