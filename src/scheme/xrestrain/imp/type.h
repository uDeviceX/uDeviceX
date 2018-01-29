enum {
    RSTR_NONE,
    RSTR_COL,
    RSTR_RBC
};

struct Restrain {
    int kind;

    /* pinned */
    int *n;
    float3 *v;
};
