#ifdef TRI_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace tri_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace tri_dev {
  #define END }
#endif

BEGIN

_S_ void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
_S_ int less(double *a, double *b) { return (*a) < (*b); }
_S_ void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}

_I_ double area_kahan(double a, double b, double c) {
    sort3(&c, &b, &a); /* a > b > c */
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
