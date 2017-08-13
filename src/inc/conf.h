// solvent

#ifndef gamma_dot
  #define gamma_dot (0.0)
  #define shear_y (false)
  #define shear_z (false)
#endif

#ifndef doublepoiseuille
#define doublepoiseuille (false)
#endif

#ifndef pushflow
  #define pushflow (false)
#else
  #ifndef driving_force
  #define driving_force (2.0)
  #endif
#endif

#ifndef contactforces
#define contactforces (false)
#endif

#ifndef multi_solvent
#define multi_solvent (false)
#endif

#ifndef global_ids
#define global_ids (false)
#endif

// dump

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

#ifndef field_dumps
#define field_dumps (false)
#endif

#ifndef field_freq
#define field_freq (1000)
#endif

#ifndef part_dumps
#define part_dumps (false)
#endif

#ifndef part_freq
#define part_freq (1000)
#endif

// solid 

#ifndef solids
#define solids (false)
#endif

#ifndef solid_mass
#define solid_mass dpd_mass
#endif

#ifndef pin_com
#define pin_com (false)
#endif

#ifndef pin_axis
#define pin_axis (false)
#endif

#ifndef sbounce_back
#define sbounce_back (false)
#endif

#ifndef rescue_freq
#define rescue_freq (100)
#endif

#ifndef pushsolid
#define pushsolid (false)
#endif

#ifndef fsiforces
#define fsiforces (false)
#endif

// spdir: [s]olid [p]eriodic [dir]ection
// example: an open cylinder along z is periodic along z, so spdir = 2
#ifdef spdir
#undef  pin_com
#define pin_com (true)
#undef  pin_axis
#define pin_axis (true)
#endif


// rbc

#ifndef rbcs
#define rbcs (false)
#endif

#ifndef RBCnv
#define RBCnv (498)
#endif

#ifndef rbounce_back
#define rbounce_back (false)
#endif

#ifndef pushrbc
#define pushrbc (false)
#endif

/* maximum allowed degree of vertex in triangulated mesh */
#define RBCmd 7

// walls

#ifndef walls
#define walls (false)
#endif

#ifndef wall_creation
#define wall_creation (1000)
#endif

// restart

#ifndef RESTART
#define RESTART (false)
#endif

#ifndef BASE_STRT_DUMP 
#define BASE_STRT_DUMP "."
#endif

#ifndef BASE_STRT_READ
#define BASE_STRT_READ "."
#endif

#ifndef strt_dumps
#define strt_dumps (false)
#endif

#ifndef strt_freq
#define strt_freq (1000)
#endif

// time
#ifndef tend
#define tend (10)
#endif

// debug
/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE) && !defined(KL_TRACE)  && \
    !defined(KL_PEEK)    && !defined(KL_UNSAFE) && \
    !defined(KL_TRACE_PEEK)
  #define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif

/* [te]xture macros */
#if !defined(TE_RELEASE) && !defined(TE_TRACE)
  #define TE_RELEASE
#endif
