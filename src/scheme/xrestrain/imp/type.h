enum {
    RSTR_NONE,
    RSTR_COL,
    RSTR_RBC
};

struct Restrain {
    int kind;
    int freq;
    
    /* pinned */
    int *n;
    float3 *v;
};
