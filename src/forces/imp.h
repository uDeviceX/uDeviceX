namespace forces {
#include "imp/type.h"
#include "imp/util.h"
#ifndef multi_solvent
  #error multi_solvent is undefined
#endif
#if    multi_solvent==true
  #include "imp/color/col2par.h"
#else
  #include "imp/grey/col2par.h"
#endif
#include "imp/main.h"

#if    defined(DPD_CHARGE)
  #include "imp/charge/main.h"
#elif  defined(DPD_GRAVITY)
  #include "imp/gravity/main.h"
#else
  #error DPD_* is undefined
#endif

}
