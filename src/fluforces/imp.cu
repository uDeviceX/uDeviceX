#include <mpi.h>
#include <assert.h>

#if __CUDACC_VER_MAJOR__ >= 9
#include <cuda_fp16.h>
#endif

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "math/rnd/imp.h"
#include "math/rnd/dev.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "cloud/imp.h"
#include "parray/imp.h"

#include "d/api.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "utils/msg.h"
#include "mpi/wrapper.h"
#include "frag/imp.h"

#include "flu/type.h"
#include "bulk/imp.h"
#include "halo/imp.h"

#include "imp.h"

/* implementation */

#include "imp/type.h"
namespace fluforces_dev {
#include "dev/main.h"
}
#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/main.h"
