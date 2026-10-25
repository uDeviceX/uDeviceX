enum {DIAG_PART_STAMP_GOOD = 123456};
struct DiagPart {
    int stamp;
    char path[FILENAME_MAX];
};
