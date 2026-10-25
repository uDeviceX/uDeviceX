#pragma once

struct BopData;
enum { POINT_N_MAX = 100 };

struct IOPointConf {
    int nn[POINT_N_MAX];
    char keys[POINT_N_MAX][FILENAME_MAX];
    int i;
};

enum { POINT_UNSET = - 1 };
struct IOPoint {
    int maxn;
    int n;

    int nkey;
    int cum_n;
    int nn[POINT_N_MAX];
    int seen[POINT_N_MAX];
    char keys[POINT_N_MAX][FILENAME_MAX];
    
    char path[FILENAME_MAX];
    BopData *bop;
};
