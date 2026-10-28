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

#ifndef D2D
#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault
#endif

#ifndef cD2D
#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))
#endif

#ifndef aD2D
#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))
#endif

#ifndef Dalloc
#define Dfree(D) CC(d::Free(D))
#define Dalloc(D, n) CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))
#define Dset(P, v, n) CC(d::Memset(P, v, (n) * sizeof(*(P))))
#define Dzero(P, n) Dset(P, 0, n)
#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n) * sizeof(*(P))))
#define DzeroA(P, n) DsetA(P, 0, n)
#endif

#ifndef EMALLOC
#define EMALLOC(n, ppdata) UC(emalloc((n) * sizeof(**(ppdata)), (void **)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void *)(dest), (const void *)(src), (n) * sizeof(*(dest)))
#endif

#ifndef BPC
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)
#endif
