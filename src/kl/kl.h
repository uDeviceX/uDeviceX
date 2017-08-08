/* TODO: header includes header */
#include "kl/common.h"
#ifdef KL_RELEASE
  #include "kl/release.h"
#elif  KL_TRACE
  #include "kl/trace.h"
#elif  KL_PEEK
  #include "kl/peek.h"
#endif
