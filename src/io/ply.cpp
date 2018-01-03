#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "inc/type.h"
#include "inc/def.h"
#include "utils/msg.h"
#include "ply.h"

#include "utils/error.h"
#include "utils/imp.h"

#ifdef PLY_WRITE_ASCII
  #include "io/ply/ascii.h"
#else
  #include "io/ply/bin.h"
#endif
#include "io/ply/common.h"
