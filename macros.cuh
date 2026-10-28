#pragma once

#ifndef ceiln
#define ceiln(m, n) (((m) + (n) - 1) / (n))
#endif

#ifndef k_cnf
#define k_cnf(n) ceiln((n), 128), 128
#endif

#ifndef dSync
#define dSync() CC(d::DeviceSynchronize())
#endif
