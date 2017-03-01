namespace Cont {
  int (*indices)[3] = NULL;
  int nt = -1;
  int nc = 0;
  struct TransformedExtent {
    float com[3];
    float transform[4][4];
  };
}
