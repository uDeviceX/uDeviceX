# preparing wall to [qw]i[t]ification

## wall.decl
Logistic::KISS* trunk
int w_n
float4 *w_pp
x::Clist *wall_cells

## k/wall.h
texWallParticles
texWallCellStart

## wall functions called by sim::
wall::interactions(SOLID_TYPE, s::pp, s::npp, s::ff)
wall::bounce(o::pp, o::n)
o::n = wall::init(o::pp, o::n)
wall:close
