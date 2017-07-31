namespace mdstr {
namespace sub {

// ini
void gen_ne(MPI_Comm cart, /**/ int* rnk_ne, int* ank_ne);

// imp
void waitall(MPI_Request rr[26]);
void cancelall(MPI_Request rr[26]);

void get_reord(const float *rr, int nm, /**/ int *reord[27], int counts[27]);
void post_sendc(const int counts[27], MPI_Comm cart, int btc, int rnk_ne[27], /**/ MPI_Request sreqc[26]);
void post_recvc(MPI_Comm cart, int btc, int ank_ne[27], /**/ int rcounts[27], MPI_Request rreqc[26]);

} // sub
} // mdstr
