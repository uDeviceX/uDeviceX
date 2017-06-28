// solvent

#ifndef doublepoiseuille
#define doublepoiseuille (false)
#endif

#ifndef pushflow
#define pushflow (false)
#endif

#ifndef contactforces
#define contactforces (false)
#endif

// dump

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

#ifndef field_dumps
#define field_dumps (false)
#endif

#ifndef part_dumps
#define part_dumps (false)
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

/* maximum allowed degree of vertex in triangulated mesh */
#define RBCmd 7

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
