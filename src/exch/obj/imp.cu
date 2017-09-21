#include <mpi.h>
#include <assert.h>

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/dev/common.h"

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "frag/imp.h"
#include "mpi/basetags.h"
#include "comm/imp.h"


#include "map.h"
#include "type.h"
#include "imp.h"

namespace exch {
namespace obj {
using namespace comm;

#include "imp/type.h"
#include "dev.h"
#include "map.h"
#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/map.h"
#include "imp/pack.h"
#include "imp/com.h"
#include "imp/unpack.h"

} // obj
} // exch
