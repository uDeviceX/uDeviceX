/* DOMAIN */
#define XS                      64
#define YS                      52
#define ZS                      56
#define XWM                     6
#define YWM                     6
#define ZWM                     6
#define XBBM                    1.f
#define YBBM                    1.f
#define ZBBM                    1.f

/* DPD */
#define numberdensity           10
#define kBT                     0.00444302
#define dt                      1e-3
#define dpd_mass                1.0
#define rbc_mass                0.5
#define solid_mass              1.0

#define adpd_b         2.6
#define adpd_r         2.6
#define adpd_br        2.6

#define gdpd_b         1.0
#define gdpd_r         5.0
#define gdpd_br        3.0


/* FEATURES */
#define rbcs                    false
#define multi_solvent           true
#define color_freq              500
#define contactforces           true
#define ljsigma                 0.3
#define ljepsilon               0.44
#define fsiforces               true
#define walls                   true
#define wall_creation           1000
#define tend                    1000000

/* FLOW TYPE */
#define pushflow                true
#define pushrbc                 true
#define driving_force           0.001

/* DUMPS */
#define dump_all_fields         true
#define part_freq               20000
#define field_dumps             true
#define field_freq              20000
#define strt_dumps              true
#define strt_freq               20000

/* Part II (added by tools/argp) */

#undef            adpd_b
#undef           adpd_br
#undef            adpd_r
#undef       DPD_GRAVITY
#undef                dt
#undef    FORCE_CONSTANT
#undef       force_dumps
#undef       FORCE_PAR_A
#undef            gdpd_b
#undef           gdpd_br
#undef            gdpd_r
#undef        part_dumps
#undef   RBC_PARAMS_LINA
#undef              rbcs
#undef           RESTART
#undef              VCON
#undef  VCON_ADJUST_FREQ
#undef     VCON_LOG_FREQ
#undef  VCON_SAMPLE_FREQ
#undef           VCON_VX
#undef           VCON_VY
#undef                XS
#undef                YS
#undef                ZS

#define           adpd_b    (2.66667)    /* */
#define          adpd_br    (2.66667)    /* */
#define           adpd_r    (2.66667)    /* */
#define      DPD_GRAVITY       (true)    /* */
#define               dt       (1e-4)    /* */
#define   FORCE_CONSTANT       (true)    /* */
#define      force_dumps       (true)    /* */
#define      FORCE_PAR_A          (0)    /* */
#define           gdpd_b       (11.5)    /* */
#define          gdpd_br       (8.25)    /* */
#define           gdpd_r          (5)    /* */
#define       part_dumps       (true)    /* */
#define  RBC_PARAMS_LINA       (true)    /* */
#define             rbcs       (true)    /* */
#define          RESTART       (true)    /* */
#define VCON_ADJUST_FREQ       (1000)    /* */
#define    VCON_LOG_FREQ        (500)    /* */
#define VCON_SAMPLE_FREQ          (1)    /* */
#define             VCON       (true)    /* */
#define          VCON_VX   (0.398015)    /* */
#define          VCON_VY (-0.0398015)    /* */
#define               XS         (64)    /* */
#define               YS         (24)    /* */
#define               ZS         (12)    /* */
