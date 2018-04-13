#ifdef FORCE_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_dev {
  #define END }
#endif

BEGIN

_I_ double f() { return 0.0; }

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
