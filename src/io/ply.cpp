#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "inc/type.h"
#include "inc/def.h"
#include "msg.h"
#include "ply.h"

#ifdef PLY_WRITE_ASCII
  #include "io/ply/ascii.h"
#else
  #include "io/ply/bin.h"
#endif
#include "io/ply/common.h"
