#include <stdio.h>

void print(int n, int x, int y, int z, bool active) {
    //printf("%2d  %c  %+d %+d %+d\n", n, active ? '+' : ' ', x, y, z);
    if (active) printf("%2d  %+d %+d %+d\n", n, x, y, z);
}

bool smaller(int i, int j, int k,
             int i_, int j_, int k_) {
    return
        (i < i_)  ||
        (i == i_ && j < j_) ||
        (i == i_ && j == j_ && k < k_);    
}

bool pred(int i, int j, int k) {
    int i_, j_, k_;
    i_ = 2-i;
    j_ = 2-j;
    k_ = 2-k;

    // always keep bulk
    if (i_ == i && j_ == j && k_ == k)
        return true;

     return smaller(i, j, k, i_, j_, k_);
    // return smaller(i_, j_, k_, i, j, k);
}

void walk() {
    int n;
    int i, j, k;
    int x, y, z;
    n = 0;
    for (k = 0; k < 3; ++k) {
        z = k - 1;
        for (j = 0; j < 3; ++j) {
            y = j - 1;
            for (i = 0; i < 3; ++i) {
                x = i - 1;
                print(n++, x, y, z, pred(i, j, k));
            }
        }
    }
}

int main() {
    walk();
    return 0;
}
