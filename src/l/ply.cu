#include <stdio.h>
#include <cassert>
#include "inc/type.h"
#include "common.h"
#include "l/ply.h"

#ifdef PLY_WRITE_ASCII
  #include "l/h/ply.ascii.h"
#else
  #include "l/h/ply.bin.h"
#endif
#include "l/h/ply.h"
