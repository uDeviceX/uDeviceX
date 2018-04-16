struct float3;
namespace force_kantor0_hst {
// tag::interface[]
float3 dih_a(float phi, float kb, float3 a, float3 b, float3 c, float3 d);
float3 dih_b(float phi, float kb, float3 a, float3 b, float3 c, float3 d);

double3 dih_a_dbl(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
double3 dih_b_dbl(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
// end::interface[]
}
