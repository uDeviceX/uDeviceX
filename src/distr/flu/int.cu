#include <mpi.h>

#include "inc/type.h"
#include "mpi/basetags.h"
#include "comm/imp.h"

using namespace comm;

namespace distr {
namespace flu {
#include "map.h"
#include "imp.h"
} // flu
} // distr

#include "int.h"
