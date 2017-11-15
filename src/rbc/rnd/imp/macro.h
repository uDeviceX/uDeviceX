#define CU(x)                                                           \
    do {                                                                \
        curandStatus_t _curc = (x);                                     \
        if ( _curc != CURAND_STATUS_SUCCESS)                            \
            ERR("curand error status: %d", _curc);                      \
    } while(0)
