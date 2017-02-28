namespace m { /* MPI (man MPI_Cart_get) */
  extern int d;
  extern int rank, coords[], dims[], periods[];
  extern bool reorder;
  extern MPI_Comm cart;
}
