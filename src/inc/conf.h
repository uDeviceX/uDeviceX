/* DPD kernel envelop parameter: random and dissipative kernels (wd = wr^2)
   0: wr = 1 - r
   1: wr = (1 - r)^(1/2)
   2: wr = (1 - r)^(1/4) */
#ifndef S_LEVEL
  #define S_LEVEL (2)
#endif

/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_TRACE)  && \
    !defined(KL_PEEK)       && !defined(KL_UNSAFE) && \
    !defined(KL_TRACE_PEEK) && !defined(KL_NONE)   && \
    !defined(KL_CPU)        && !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC) && !defined(CC_TRACE) && !defined(CC_TRACE_PEEK)
  #define CC_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

#ifndef BASE_STRT_DUMP
#define BASE_STRT_DUMP "strt"
#endif

#ifndef BASE_STRT_READ
#define BASE_STRT_READ "strt"
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

// walls
#ifndef walls
#define walls (false)
#endif

/* ids for cell */
#ifndef rbc_ids
#define rbc_ids (false)
#endif

#ifndef force_dumps
#define force_dumps (false)
#endif

#ifndef rbc_com_dumps
  #define rbc_com_dumps (false)
#endif

/* stretch cell?  see doc/stretch.md */
#ifndef RBC_STRETCH
  #define RBC_STRETCH (false)
#endif

/* maximum allowed degree of a vertex */
#define RBCmd 7

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

#ifndef empty_solid_particles
#define empty_solid_particles (true)
#endif

/* recolor solvent crossing periodic boundary ? */
#ifdef RECOLOR_FLUX
#error RECOLOR_FLUX is runtime: recolor.active
#endif
#ifdef COL_FLUX_DIR
#error COL_FLUX_DIR is runtime: recolor.dir
#endif



#if defined(RESTART)
  #error RESTART is runtime: glb.restart
#endif

#if defined(freq_strt)
  #error freq_strt is runtime: dump.freq_strt
#endif

#if defined(strt_dumps)
  #error strt_dumps is runtime dump.strt
#endif

#if defined(kBT)
  #error kBT is runtime: glb.kBT
#endif

#if defined(RBCnv)
  #error RBCnv is runtime: read from rbc.off
#endif

#if defined(RBCtotArea)
  #error RBCtotArea is runtime: set rbc.totArea
#endif

#if defined(RBCtotVolume)
  #error RBCtotVolume is runtime: set rbc.totVolume
#endif

#if defined(wall_creation)
  #error wall_creation is runtime: time.wall
#endif

#if defined(dt)
  #error dt is runtime: time.dt
#endif

#if defined(tend)
  #error tend is runtime: time.end
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
#ifdef contactforces
#error cnt is runtime
#endif

/* should solvent have colors? 
   see doc/color.md */
#ifdef multi_solvent
#error multi_solvent is runtime: flu/colors in cfg
#endif

/* when to re-color rbcs */
#ifdef color_freq
#error color_freq is runtime: flu.recolor_freq (int)
#endif

/* ids for solvent */
#ifdef global_ids
#error  global_ids is runtime: flu/ids in cfg
#endif


#ifdef RBC_RND
    #error RBC_RND is runtime: rbc.rnd
#endif


// dpd

#ifdef flu_mass
#error flu_mass is runtime: flu.mass
#endif


// solid

#ifdef solids
#error solids: flag is runtime now, rig.active in cfg
#endif

#ifdef solid_mass
#error solid_mass is runtime: rig.mass
#endif

#if defined(pin_com) || defined(pin_comx) || defined(pin_comy) || \
    defined(pin_comz) || defined(pin_axis)
#error pin info are runtime
#endif

#ifdef sbounce_back
#error sbounce_back is runtime: rig/sbounce in cfg
#endif

#ifdef pushflu
#error pushflu is runtime: flu.push
#endif

#ifdef pushsolid
#error pushsolid is runtime: rig.push
#endif

#ifdef fsiforces
#error fsi is runtime
#endif

#ifdef spdir
#error spdir is runtime
#endif

// rbc
#ifdef rbcs
#error rbcs is runtime: rbc/active in cfg
#endif

#ifdef rbc_mass
#error rbc_mass is runtime: rbc.mass
#endif

#ifdef pushrbc
#error pushrbc is runtime: rbc.push
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

#ifdef RBC_STRESS_FREE
#error RBC_STRESS_FREE is runtime: rbc.stress_free
#endif

#if defined(adpd_b) || defined(adpd_br) || defined(adpd_r)
#error adpd* is runtime
#endif

#if defined(gdpd_b) || defined(gdpd_br) || defined(gdpd_r)
#error gdpd* is runtime
#endif

#if defined(ljsigma) || defined(ljepsilon)
#error lj* is runtime
#endif

