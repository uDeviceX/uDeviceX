#define CU(x)                                                           \
    do {                                                                \
        int _curc = (x);                                                \
        if ( _curc != 0)                                                \
            ERR("curand error status: %d", _curc);                      \
    } while(0)
