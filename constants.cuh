#ifndef UDX_CONSTANTS_CUH
#define UDX_CONSTANTS_CUH


#define XMACRO_COLOR(_)                         \
    _(BLUE)                                     \
    _(RED)

#define ENUM_COLOR(a) a##_COLOR
#define make_enum_color(a) ENUM_COLOR(a) ,
enum {
    XMACRO_COLOR(make_enum_color)
    N_COLOR
};
#undef make_enum_color

enum {

    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,


    MAX_PSOLID_NUM = 12000,


    MAX_SOLIDS = 100,


    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,


    MAX_OBJ_DENSITY = 30,


    MAX_CELL_NUM = 1500,


    MAX_TEXO_SIZE = 100000000,


    HSAFETY_FACTOR = 10,


    ODSTR_FACTOR = 3,
};


#define RBC_SPRING_CAP (0.99)

#endif
