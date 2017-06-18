namespace l { namespace ply {
static bool prop(const char *pname, const char *str)
{
  const int l1 = strlen(pname);
  const int l2 = strlen(str);

  if (l1 > l2) return false;
  for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
  return true;
}

void read(const char *fname, Mesh *m)
{
  FILE *f = fopen(fname, "r");

  if (f == NULL)
    {
      fprintf(stderr, "error. could not read <%s>\nexiting...\n", fname);
      exit(1);
    }

  int l = 0;
  m->nt = m->nv = -1;

#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

  // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s

  while (l++ < MAXLINES)
    {
      char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

      const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);

      if (checker != 1)
	{
	  fprintf(stderr, "Something went wrong reading <%s>\n", fname);
	  exit(1);
	}

      int ibuf;
      if (sscanf(cbuf, "element vertex %d", &ibuf) == 1) m->nv = ibuf;
      else if (sscanf(cbuf, "element face %d", &ibuf) == 1) m->nt = ibuf;
      else if (prop("end_header", cbuf)) break;
    }

  if (l >= MAXLINES || m->nt == -1 || m->nv == -1)
    {
      printf("Something went wrong, did not catch end_header\n");
      exit(1);
    }

  m->tt = new int[3 * m->nt];
  m->vv = new float[3 * m->nv];

  for (int i = 0; i < m->nv; ++i)
    fscanf(f, "%f %f %f\n",
	   m->vv + 3*i + 0,
	   m->vv + 3*i + 1,
	   m->vv + 3*i + 2);

  for (int i = 0; i < m->nt; ++i)
    fscanf(f, "%*d %d %d %d\n",
	   m->tt + 3*i + 0,
	   m->tt + 3*i + 1,
	   m->tt + 3*i + 2);

  fclose(f);
}
}}
