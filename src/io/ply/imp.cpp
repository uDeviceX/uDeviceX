#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "inc/type.h"
#include "inc/def.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "utils/imp.h"


#include "imp.h"

#ifdef PLY_WRITE_ASCII
  #include "imp/ascii.h"
#else
  #include "imp/bin.h"
#endif
