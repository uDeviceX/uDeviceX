#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "inc/type.h"
#include "inc/def.h"
#include "msg.h"
#include "l/ply.h"

#ifdef PLY_WRITE_ASCII
  #include "l/h/ply.ascii.h"
#else
  #include "l/h/ply.bin.h"
#endif
#include "l/h/ply.h"
