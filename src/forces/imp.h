namespace forces {
#include "imp/util.h"
#include "imp/main.h"

#if    defined(DPD_CHARGE)
  #include "charge/main.h"
#elif  defined(DPD_GRAVITY)
  #include "gravity/main.h"
#else
  #error DPD_* is undefined
#endif

}
