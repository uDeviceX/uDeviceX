namespace Cont {
  int (*indices)[3] = NULL;
  struct TransformedExtent {
    float com[3];
    float transform[4][4];
  };
}
