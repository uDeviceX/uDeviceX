#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "utils/cc.h"
#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"
#include "utils/error.h"

#include "math/rnd/imp.h"
#include "math/rnd/dev.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "transpose/imp.h"

#include "utils/kl.h"
#include "forces/type.h"
#include "forces/pack.h"
#include "forces/imp.h"
#include "imp.h"

#include "imp/type.h"

#include "dev/float.h"
#include "dev/decl.h"
#include "dev/pack.h"
namespace asmb {
#include "dev/asm.h"
}
#include "dev/fetch.h"
#include "cloud/lforces/get.h"
#include "dev/dpd.h"
#include "dev/core.h"

#define MYCPBX  (4)
#define MYCPBY  (2)
#define MYCPBZ  (2)
#define MYWPB   (4)

#include "dev/merged.h"
#include "dev/tex.h"
#include "cloud/lforces/int.h"

#include "imp/setup.h"
#include "imp/tex.h"
#include "imp/info.h"
#include "imp/main.h"
