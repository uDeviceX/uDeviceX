enum {MAX_NK = 100 };

struct KeyList {
    int nk; /* number of keys */
    int  ww[MAX_NK]; /* width */
    int  mark[MAX_NK];
    char keys[MAX_NK][FILENAME_MAX];
};
