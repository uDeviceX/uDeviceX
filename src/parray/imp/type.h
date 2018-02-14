struct PaArray {
    bool colors;
    union {
        PaArray_v pa;
        PaCArray_v pac;
    };
}
