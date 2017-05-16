#include <mpi.h>

/* [B]ase [T]ags for mpi messages
 * C: [c]ounts
 * P: [p]articles
 * S: [s]olid
 * CS: [C]ell [S]tart
 */

#define STRIDE (1000)
enum 
{
    /* dpd.impl.h */
    BT_P_DPD = 0,
    BT_P2_DPD = BT_P_DPD  + STRIDE,
    BT_CS_DPD = BT_P2_DPD + STRIDE,
    BT_C_DPD = BT_CS_DPD  + STRIDE,

    /* sdstr.impl.h */
    BT_C_SDSTR = BT_C_DPD    + STRIDE,
    BT_P_SDSTR = BT_C_SDSTR  + STRIDE,
    BT_P2_SDSTR = BT_P_SDSTR + STRIDE,

    /* wall.impl.h (init) */
    BT_C_WALL = BT_P2_SDSTR + STRIDE,
    BT_P_WALL = BT_C_WALL   + STRIDE,

    /* rex.impl.h */
    BT_C_REX = BT_P_WALL + STRIDE,
    BT_P_REX = BT_C_REX  + STRIDE,
    BT_P2_REX = BT_P_REX + STRIDE,
    BT_A_REX = BT_P2_REX + STRIDE,

    /* rdstr.impl.h */
    BT_C_RDSTR = BT_A_REX   + STRIDE,
    BT_P_RDSTR = BT_C_RDSTR + STRIDE,

    /* bbhalo.impl.h */
    BT_C_BBHALO = BT_P_RDSTR   + STRIDE,
    BT_S_BBHALO = BT_C_BBHALO  + STRIDE,
    BT_P_BBHALO = BT_S_BBHALO  + STRIDE,
    BT_S2_BBHALO = BT_P_BBHALO + STRIDE,
    
};
#undef STRIDE
