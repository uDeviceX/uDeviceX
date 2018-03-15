enum { N_MAX = 100 };

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

struct IOPoint {
    int maxn;
    int i;
    int nn[N_MAX];
    char key[N_MAX*FILENAME_MAX];
    char path[FILENAME_MAX];
};
