#pragma once

/* One build policy for every compilation unit in the executable. */
#if defined(DEV_CUDA) && defined(DEV_CPU)
#error "DEV_CUDA and DEV_CPU cannot both be defined"
#endif
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
#define DEV_CUDA
#endif

#if !defined(CC_RELEASE) && !defined(CC_SYNC)
#define CC_RELEASE
#endif
#if defined(CC_RELEASE) && defined(CC_SYNC)
#error "Select one CUDA error-checking mode"
#endif

#if !defined(KL_RELEASE) && !defined(KL_NONE) && !defined(KL_SYNC)
#define KL_RELEASE
#endif
#if (defined(KL_RELEASE) + defined(KL_NONE) + defined(KL_SYNC)) != 1
#error "Select one kernel-launch mode"
#endif

#if !defined(CUDA_ALLOC_RELEASE) && !defined(CUDA_ALLOC_DEBUG)
#define CUDA_ALLOC_RELEASE
#endif
#if defined(CUDA_ALLOC_RELEASE) && defined(CUDA_ALLOC_DEBUG)
#error "Select one CUDA allocation mode"
#endif

#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
#define RBC_DOUBLE
#endif
#if defined(RBC_DOUBLE) && defined(RBC_FLOAT)
#error "Select one RBC precision"
#endif

#if !defined(RBC_SPRING_IGNORE) && !defined(RBC_SPRING_FAIL)
#define RBC_SPRING_IGNORE
#endif
#if defined(RBC_SPRING_IGNORE) && defined(RBC_SPRING_FAIL)
#error "Select one RBC spring policy"
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

#include "api.cuh"
