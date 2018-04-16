#ifdef FORCE_KANTOR1_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor1_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor1_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR1_HOST
_S_ double rsqrt0(double x) { return pow(x, -0.5); }
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif


_I_ double3 dih_a(double phi, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 ans;
    ans.x = ans.y = ans.z = 0;
    return ans;    
}

_I_ double3 dih_b(double phi, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 ans;
    ans.x = ans.y = ans.z = 0;
    return ans;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
