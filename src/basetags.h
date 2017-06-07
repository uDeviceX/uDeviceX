
/* [B]ase [T]ags for mpi messages
 * C: [c]ounts
 * P: [p]articles
 * S: [s]olid
 * CS: [C]ell [S]tart
 */

#define STRIDE (100)

#define tag_list(_)                             \
    /* dpd.impl.h */                            \
    _(BT_P_DPD)                                 \
    _(BT_P2_DPD)                                \
    _(BT_CS_DPD)                                \
    _(BT_C_DPD)                                 \
    /* odstr.impl.h */                          \
    _(BT_C_ODSTR)                               \
    _(BT_P_ODSTR)                               \
    _(BT_P2_ODSTR)                              \
    /* wall.impl.h (init) */                    \
    _(BT_C_WALL)                                \
    _(BT_P_WALL)                                \
    /* rex.impl.h */                            \
    _(BT_C_REX)                                 \
    _(BT_P_REX)                                 \
    _(BT_P2_REX)                                \
    _(BT_A_REX)                                 \
    /* sdstr.impl.h */                          \
    _(BT_C_SDSTR)                               \
    _(BT_P_SDSTR)                               \
    _(BT_S_SDSTR)                               \
    /* bbhalo.impl.h */                         \
    _(BT_C_BBHALO)                              \
    _(BT_S_BBHALO)                              \
    _(BT_P_BBHALO)                              \
    _(BT_S2_BBHALO)                             \
    /* rdstr.impl.h */                          \
    _(BT_C_RDSTR)                               \
    _(BT_P_RDSTR)


#define make_id_name(a) a##_ID,
#define make_tag_val(a) a = STRIDE * a##_ID,

enum tag_ids {
    tag_list(make_id_name)
};

enum {
    tag_list(make_tag_val)
};

#undef STRIDE
#undef tag_list
#undef make_id_name
#undef make_tag_val
