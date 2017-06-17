namespace m { /* MPI (man MPI_Cart_create) */
extern const int d;
extern int rank, coords[], dims[], periods[];
extern const bool reorder;
extern MPI_Comm cart;
}
