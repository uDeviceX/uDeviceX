namespace l { namespace ply {{
void write(const char *fname, const Mesh m)
{
  int i;
  FILE * f = fopen(fname, "w");

  assert(f != NULL);

  fprintf(f, "ply\n");
  fprintf(f, "format ascii 1.0\n");
  fprintf(f, "element vertex %d\n", m.nv);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  fprintf(f, "element face %d\n", m.nt);
  fprintf(f, "property list int int vertex_index\n");
  fprintf(f, "end_header\n");

  for (i = 0; i < m.nv; ++i)
    fprintf(f, "%f %f %f\n", m.vv[3*i + 0], m.vv[3*i + 1], m.vv[3*i + 2]);

  for (i = 0; i < m.nt; ++i)
    fprintf(f, "3 %d %d %d\n", m.tt[3*i + 0], m.tt[3*i + 1], m.tt[3*i + 2]);

  fclose(f);
}
}}
