/** Copyright 1993-2013 NVIDIA Corporation.  All rights reserved. **/

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
