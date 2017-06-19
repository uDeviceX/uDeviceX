namespace Cont {
template <typename T>
void remove(T* data, int nv, int *e, int nc) {
  int c; /* c: cell, v: vertex */
  for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}
}
