#include <limits>
#include <stdint.h>
#include <stdio.h>
#include "rnd.h"
#include "common.h"
#include "common.cuda.h"
#include "inc/type.h"
#include "dpd/local.h"
#include <conf.h>
#include "forces.h"

#include "dpd/imp/type.h"
#include "dpd/dev/float.h"
#include "dpd/dev/decl.h"

#define __IMOD(x,y) ((x)-((x)/(y))*(y))

#include "dpd/dev/pack.h"
#include "dpd/dev/dpd.h"
#include "dpd/dev/core.h"

#define MYCPBX  (4)
#define MYCPBY  (2)
#define MYCPBZ  (2)
#define MYWPB   (4)

#include "dpd/dev/merged.h"
#include "dpd/dev/tex.h"
#include "dpd/dev/transpose.h"

#include "dpd/imp/decl.h"
#include "dpd/imp/setup.h"
#include "dpd/imp/tex.h"
#include "dpd/imp/info.h"
#include "dpd/imp/flocal.h"
