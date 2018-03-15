#define N_MAX 100

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

struct IOPoint {
    int maxn;
    IOPointConf c;
    char path[FILENAME_MAX];
};
