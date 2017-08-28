#if   defined(DBG_NONE)
  #include "dbg/macro/none.h"
#elif defined(DBG_PEEK)
  #include "dbg/macro/peek.h"
#elif defined(DBG_SILENT)
  #include "dbg/macro/silent.h"
#elif defined(DBG_TRACE)
  #include "dbg/macro/trace.h"
#else
  #error DBG_* is undefined
#endif

#include "dbg/macro/common.h"
