#if defined(dt)
  #error dt is runtime
#endif

#if defined(XS) || defined(YS) || defined(ZS)
  #error Subdomain sizes are runtime
#endif

#if defined(field_dumps) || defined (field_freq) ||     \
    defined(part_dumps) || defined (part_freq)
#error field and part dumps are runtime!
#endif

#if defined(VCON) || defined(VCON_CART) || defined (VCON_RAD) || \
    defined(VCON_VX) || defined(VCON_VY) || defined(VCON_VZ)  || \
    defined(VCON_FACTOR) || defined(VCON_SAMPLE_FREQ) || \
    defined(VCON_ADJUST_FREQ) || defined(VCON_LOG_FREQ)
#error vcon has runtime parameters now!
#endif

#if defined(WVEL_PAR_A) || defined(WVEL_PAR_U) || defined(WVEL_PAR_U) || \
    defined(WVEL_PAR_H) || defined(WVEL_PAR_W) || defined(WVEL_PAR_Z) || \
    defined(WVEL_PAR_Y) || defined(WVEL_SIN) || defined(WVEL_PAR_LOG_FREQ)
#error wvel is runtime now!
#endif

#if defined(XWM) || defined(YWM) || defined(ZWM)
#error [XYZ]W is set in inc/def.h and should not be set in conf.h
#endif

#if defined(FORCE_PAR_A)
#error bforce has runtime parameters now!
#endif

/* object-object, cell-object, and cell-cell contact force */
#ifndef contactforces
#define contactforces (false)
#endif

/* should solvent have colors? 
   see doc/color.md */
#ifndef multi_solvent
#define multi_solvent (false)
#endif

/* when to re-color rbcs */
#ifndef color_freq
#define color_freq    (0)
#endif

/* recolor solvent crossing periodic boundary ? */
#ifndef RECOLOR_FLUX
#define RECOLOR_FLUX (false)
#endif
#ifndef COL_FLUX_DIR
#define COL_FLUX_DIR (0)
#endif

/* ids for solvent */
#ifndef global_ids
#define global_ids (false)
#endif

/* ids for cell */
#ifndef rbc_ids
#define rbc_ids (false)
#endif

// dump
#ifndef dump_all_fields
#define dump_all_fields (false)
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

#ifndef force_dumps
#define force_dumps (false)
#endif

#ifndef rbc_com_dumps
  #define rbc_com_dumps (false)
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

/* stretch cell?  see doc/stretch.md */
#ifndef RBC_STRETCH
  #define RBC_STRETCH (false)
#endif

#ifndef RBC_RND
   #define RBC_RND (false)
#endif

/* dump meshes relative to the domain edge or domain center? */
#if !defined(MESH_SHIFT_EDGE) && !defined(MESH_SHIFT_CENTER)
  #define MESH_SHIFT_EDGE
#endif

/* assert */
#if rbc_com_dumps && !rbc_ids
    #error "Need rbc ids for rbc_com_dumps"
#endif

#ifndef rbc_com_freq
#define rbc_com_freq (1000)
#endif


// dpd

#ifndef flu_mass
#define flu_mass 1.0
#endif


// solid

#ifndef solids
#define solids (false)
#endif

#ifndef solid_mass
#define solid_mass flu_mass
#endif

#ifndef pin_com
#define pin_com (false)
#endif

#ifndef pin_comx
#define pin_comx (false)
#endif

#ifndef pin_comy
#define pin_comy (false)
#endif

#ifndef pin_comz
#define pin_comz (false)
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

#ifndef pushflu
#define pushflu (true)
#endif

#ifndef pushsolid
#define pushsolid (false)
#endif

#ifndef fsiforces
#define fsiforces (false)
#endif

#ifndef empty_solid_particles
#define empty_solid_particles (true)
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

#ifndef rbc_mass
#define rbc_mass flu_mass
#endif

#ifndef RBCnv
#define RBCnv (498)
#endif
/* [n]umber of [t]riangles (Euler formula) */
#define RBCnt ( 2*(RBCnv) - 4 )
/* maximum allowed degree of a vertex */
#define RBCmd 7

#ifndef rbounce_back
#define rbounce_back (false)
#endif

#ifndef pushrbc
#define pushrbc (false)
#endif

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
#define BASE_STRT_DUMP "strt"
#endif

#ifndef BASE_STRT_READ
#define BASE_STRT_READ "strt"
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
/* dbg macros */
#if !defined(DBG_NONE)    && !defined(DBG_TRACE) && \
    !defined(DBG_SILENT)  && !defined(DBG_PEEK)
#define DBG_NONE
#endif

/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_TRACE)  && \
    !defined(KL_PEEK)       && !defined(KL_UNSAFE) && \
    !defined(KL_TRACE_PEEK) && !defined(KL_NONE)   && \
    !defined(KL_CPU)        && !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [m]pi [c]heck macro */
#if !defined(MC_RELEASE)
  #define MC_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC) && !defined(CC_TRACE) && !defined(CC_TRACE_PEEK)
  #define CC_RELEASE
#endif

/* [te]xture macros */
#if !defined(TE_RELEASE) && !defined(TE_TRACE)
  #define TE_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* a radius of the spherical drop */
#ifdef BANGLADESH_R
#error BANGLADESH is runtime now. see iccolor
#endif

/* make a center of mass velocity zero? */
#if defined(RESTRAIN_RED_VEL) || defined(RESTRAIN_RBC_VEL) ||   \
    defined(RESTRAIN_NONE) || defined(RESTRAIN_REPORT_FREQ)
  #error RESTRAIN in runtime now
#endif

/* RBC membrane parameter sets */
#if defined(RBC_PARAMS_TEST) || defined(RBC_PARAMS_LINA)
   #error RBC_PARAMS is runtime!
#endif

#ifndef RBC_STRESS_FREE
  #define RBC_STRESS_FREE (false)
#endif

/* DPD kernel envelop parameter: random and dissipative kernels (wd = wr^2)
   0: wr = 1 - r
   1: wr = (1 - r)^(1/2)
   2: wr = (1 - r)^(1/4) */
#ifndef S_LEVEL
  #define S_LEVEL (2)
#endif

/*** TODO ***/
#ifndef gdpd_s
  #define gdpd_s gdpd_b
#endif

#ifndef adpd_s
  #define adpd_s adpd_b
#endif
/**********/

/*** see poc/color */
#ifndef gdpd_bw
  #define gdpd_bw gdpd_b
#endif

#ifndef adpd_bw
  #define adpd_bw adpd_b
#endif

#ifndef gdpd_bs
  #define gdpd_bs gdpd_b
#endif

#ifndef adpd_bs
  #define adpd_bs adpd_b
#endif

#ifndef gdpd_rw
  #define gdpd_rw gdpd_r
#endif

#ifndef adpd_rw
  #define adpd_rw adpd_r
#endif

#ifndef gdpd_rs
  #define gdpd_rs gdpd_r
#endif

#ifndef adpd_rs
  #define adpd_rs adpd_r
#endif

#ifndef gdpd_sw
  #define gdpd_sw gdpd_b
#endif

#ifndef adpd_sw
  #define adpd_sw adpd_b
#endif
/*** */
