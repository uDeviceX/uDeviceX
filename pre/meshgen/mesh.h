#include <vector>

struct float3 {float x, y, z;};

void icosahedron(std::vector<int>& tt, std::vector<float>& vv);

void soa2aos(const std::vector<int>& tt, const std::vector<float>& vv, std::vector<float3>& aos);
void aos2soa(const std::vector<float3>& aos, std::vector<int>& tt, std::vector<float>& vv);

void subdivide2(std::vector<int>& tt, std::vector<float>& vv);
void subdivide3(std::vector<int>& tt, std::vector<float>& vv);

void scale_to_usphere(std::vector<float>& vv);
void scale(std::vector<float>& vv, const float sc);

int flip_edges(std::vector<int>& tt, const std::vector<float>& vv);
