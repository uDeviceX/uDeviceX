# preparing wall to [qw]i[t]ification

## wall.decl
Logistic::KISS* trunk
int w_n
Particle *w_pp
x::Clist *wall_cells

## wall functions called by sim::
wall::interactions(SOLID_TYPE, s::pp, s::npp, /**/ s::ff)
o::n = wall::init(o::pp, o::n)
wall:close
