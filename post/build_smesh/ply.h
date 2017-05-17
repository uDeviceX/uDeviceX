#include <vector>

void write_ply(const char *fname, const std::vector<int>& tt, const std::vector<float>& vv);
void read_ply(const char *fname, std::vector<int>& tt, std::vector<float>& vv);
