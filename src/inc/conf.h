// solvent

/* DPD kernel envelop parameter: random and dissipative kernels
   wr = (1-r)^S_LEVEL
   wd = (1-r)^(2*S_LEVEL) */
#ifndef S_LEVEL
  #define S_LEVEL (2)
#endif

#ifndef gamma_dot
  #define gamma_dot (0.0)
  #define shear_y (false)
  #define shear_z (false)
#endif

/* gamma_dot time profile */
#if !defined(GDOT_FLAT) && !defined(GDOT_DUPIRE_UP) && !defined(GDOT_DUPIRE_DOWN)
  #define GDOT_FLAT
#endif
#ifndef GDOT_REPORT_FREQ
  #define GDOT_REPORT_FREQ (1000)
#endif

#ifndef FORCE_PAR_A
#define FORCE_PAR_A (2.0)
#endif

#ifndef contactforces
#define contactforces (false)
#endif

#ifndef multi_solvent
#define multi_solvent (false)
#endif

// when to re-color rbcs
#ifndef color_freq
#define color_freq    (0)
#endif

// recoloring options

#ifndef RECOLOR_FLUX
#define RECOLOR_FLUX (false)
#endif

#ifndef COL_FLUX_DIR
#define COL_FLUX_DIR (0)
#endif

// global ids for solvent, rbcs

#ifndef global_ids
#define global_ids (false)
#endif

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

#ifndef field_dumps
#define field_dumps (false)
#endif

#ifndef field_freq
#define field_freq (1000)
#endif

#ifndef part_dumps
#define part_dumps (false)
#endif

#ifndef force_dumps
#define force_dumps (false)
#endif

#ifndef part_freq
#define part_freq (1000)
#endif

#ifndef rbc_com_dumps
#define rbc_com_dumps (false)
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

#if !defined(ODSTR0) && !defined(ODSTR1) && !defined(ODSTR_SAFE)
  #define ODSTR1
#endif

/* forces in sim:: on/off */
#if !defined(FORCE0) && !defined(FORCE1)
  #define FORCE1
#endif

/* body force */
#if !defined(FORCE_NONE) && !defined(FORCE_DOUBLE_POISEUILLE) && \
    !defined(FORCE_4ROLLER) && !defined(FORCE_CONSTANT)
  #define FORCE_NONE
#endif

/* a radius of the spherical drop */
#ifndef BANGLADESH_R
  #define BANGLADESH_R (4)
#endif

/* make a center of mass velocity zero? */
#if !defined(RESTRAIN_RED_VEL) && !defined(RESTRAIN_RBC_VEL) && \
    !defined(RESTRAIN_NONE)
  #define RESTRAIN_NONE
#endif

/* RBC membrain parameter sets */
#if !defined(RBC_PARAMS_TEST) && !defined(RBC_PARAMS_LINA)
  #define RBC_PARAMS_TEST
#endif

/* MSG frequency */
#ifndef RESTRAIN_REPORT_FREQ
  #define RESTRAIN_REPORT_FREQ (1000)
#endif

/*           Velocity controller           */

#ifndef VCON
  #define VCON (false)
#endif

#ifndef VCON_SAMPLE_FREQ
  #define VCON_SAMPLE_FREQ (10)
#endif

#ifndef VCON_ADJUST_FREQ
  #define VCON_ADJUST_FREQ (500)
#endif

#ifndef VCON_LOG_FREQ
  #define VCON_LOG_FREQ (0)
#endif

#ifndef VCON_VX
  #define VCON_VX (1.f)
#endif

#ifndef VCON_VY
  #define VCON_VY (0.f)
#endif

#ifndef VCON_VZ
  #define VCON_VZ (0.f)
#endif

#ifndef VCON_FACTOR
  #define VCON_FACTOR (0.08f)
#endif

/* how mass affects dpd forces: like "gravity" ~ mi*mj or like
   "charge" -- no dependencies on mass */
#if !defined(DPD_GRAVITY) && !defined(DPD_CHARGE)
  #define DPD_CHARGE
#endif

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
