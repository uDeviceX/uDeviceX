#!/usr/bin/awk -f

# tsdf - tiny sdf generator
#   usage: tsdf def_file sdf_file [vtk_file]
# TEST: tsdf1
# tsdf examples/ywall1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf2
# tsdf examples/ywall2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf3
# tsdf examples/sphere1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf4
# tsdf examples/sphere2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf5
# tsdf examples/cylinder1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf6
# tsdf examples/cylinder2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf7
# tsdf examples/out_sphere1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf8
# tsdf examples/out_cylinder1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf9
# tsdf examples/channel1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf10
# tsdf examples/block1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf11
# tsdf examples/block2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf12
# tsdf examples/ellipse1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf13
# tsdf examples/egg1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf14
# tsdf examples/block1rot.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf15
# tsdf examples/egg1rot.tsdf   sdf.dat sdf.out.vti
#
# TEST: tsdf16
# tsdf examples/cylinder1rot.tsdf  sdf.dat sdf.out.vti

function randint(n) { return int(rand()*n)+1 }

function init() {
    srand()

    HOME = ENVIRON["HOME"]
    processor_file = HOME "/" ".udx/processor.tmp.cpp"

    CXX="g++"
    CPPFLAGS="-O2 -g"

    if (ARGC<3) esage()

    sdf_file = ARGV[2]
    ARGV[2] = ""

    vtk_file = ARGV[3]
    ARGV[3] = ""

    TD  = ENVIRON["TD"]

}

function wsystem(cmd) {
    printf "(tsdf.awk) : running %s\n", cmd
    system(cmd)
}

function esage() {
    usage()
    EXIT_FAILURE = 1
    exit
}

function usage () {
    printf "tsdf - tiny sdf generator\n"
    printf "   usage: tsdf def_file sdf_file [vtk_file]\n"
}

function gensp(i) {
    return sprintf ("%" i "s", " ")
}

function psub(r, t) { gsub("%" r "%", t, processor) }

function csub(r, t) { gsub("//%" r "%", t, processor) } # subst in comments
function format_expr( e,     i, n, ans, tab, sep) {
    n = length(e)
    if (n==1)
	return e[1]

    tab = gensp(18)
    ans = "(\n"
    for (i = 1; i<=n; i++) {
	ans = ans sep tab "  " e[i]
	sep = ",\n"
    }
    ans = ans "\n" tab ")"
    
    return ans
}

function format_line(expr, tab, ans, update_fun) {
    update_fun = VOID_WINS ? "void_wins" : "wall_wins"
    if (INVERT)
	ans = sprintf("s = %s(s, -(%s));", update_fun, expr)
    else
	ans = sprintf("s = %s(s,    %s);", update_fun, expr)
    gsub("\n", "\n" ans)
    return tab ans
}

function add_rot(   e, m) {
    e[++m] = sprintf("rot(%s, %s, %s)", phix, phiy, phiz)
    return m
}

function add_rc_plane(e, m) { # add block center of rotation
    e[++m] = upd_def(xo, "xo", "xo = xc")
    e[++m] = upd_def(yo, "yo", "yo = yc")
    e[++m] = upd_def(zo, "zo", "zo = zc")
    return m
}
function expr_plane(     nx, ny, nz, x0, y0, z0,     m, e) {
    x0=$3; y0=$4; z0=$5
    nx=$7;  ny=$8;  nz=$9
    e[++m] =         "x  = xorg, y = yorg, z = zorg"
    if (prev_rot) {
	m = add_rc_plane(e, m)
	m = add_rot(e, m)
    }
    e[++m] = sprintf("nx = %s, ny = %s, nz = %s", nx, ny, nz)
    e[++m] = sprintf("x0 = %s, y0 = %s, z0 = %s", x0, y0, z0)     
    e[++m] = "n_abs = sqrt(nz*nz+ny*ny+nx*nx)"
    e[++m] = "(nz*(z0-z))/n_abs+(ny*(y0-y))/n_abs+(nx*(x0-x))/n_abs"
    return format_expr(e)
}

function add_rc_cylinder(e, m) { # add block center of rotation
    e[++m] = upd_def(xo, "xo", "xo = xp")
    e[++m] = upd_def(yo, "yo", "yo = yp")
    e[++m] = upd_def(zo, "zo", "zo = zp")
    return m
}
function expr_cylinder(     ) {
    ax=$3;  ay=$4;  az=$5
    xp=$7;  yp=$8;  zp=$9
    R = $11
    e[++m] =         "x  = xorg, y = yorg, z = zorg"
    if (prev_rot) {
	m = add_rc_cylinder(e, m)
	m = add_rot(e, m)
    }
    e[++m] = sprintf("ax = %s, ay = %s, az = %s", ax, ay, az)
    e[++m] = sprintf("xp = %s, yp = %s, zp = %s", xp, yp, zp)    
    e[++m] = "a2 = az*az+ay*ay+ax*ax"
    e[++m] = "D = sqrt((z-(az*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2-zp)" \
          "*(z-(az*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2-zp)" \
          "+(y-yp-(ay*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2)" \
          " *(y-yp-(ay*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2)" \
          "+(x-xp-(ax*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2)" \
          " *(x-xp-(ax*(az*(z-zp)+ay*(y-yp)+ax*(x-xp)))/a2))"
    e[++m] = sprintf("%s - D", R)
    return format_expr(e)
}

function expr_ellipse(     ax, xp, yp, zp, rx, ry, ang) {
    ax = $3 # XY, XZ, YZ
    xp=$5;  yp=$6;  zp=$7
    rx = $9; ry = $10
    ang = $12
    e[++m] =         "x  = xorg, y = yorg, z = zorg"
    if (prev_rot) {
	m = add_rc_cylinder(e, m) # like a cylinder
	m = add_rot(e, m)
    }
    e[++m] = sprintf("xp = %s, yp = %s, zp = %s", xp, yp, zp)
    e[++m] = sprintf("rx = %s, ry = %s", rx, ry)
    e[++m] = sprintf("ang = %s", ang)
    if (ax == "XY") {
	e[++m] = "x0 = sin(ang)*(y-yp)+cos(ang)*(x-xp)"
	e[++m] = "y0 = cos(ang)*(y-yp)-sin(ang)*(x-xp)"
    } else if (ax == "XZ") {
	e[++m] = "x0 = sin(ang)*(z-zp)+cos(ang)*(x-xp)"
	e[++m] = "y0 = cos(ang)*(z-zp)-sin(ang)*(x-xp)"
    } else if (ax == "YZ") {
	e[++m] = "x0 = sin(ang)*(z-zp)+cos(ang)*(y-yp)"
	e[++m] = "y0 = cos(ang)*(z-zp)-sin(ang)*(y-yp)"
    } else {
	printf "unknown ax in ellipse command (should be XY, XZ, YZ)\n" > "/dev/stderr"
	exit
    }
    e[++m] = \
	"-(1.0*(pow(y0,2.0)+pow(x0,2.0))"				\
	"*(pow(pow(rx,2.0)*pow(y0,2.0)+pow(ry,2.0)*pow(x0,2.0),0.5)-1.0*rx*ry)" \
	"*pow(pow(rx,6.0)*pow(y0,6.0)+3.0*pow(rx,4.0)*pow(ry,2.0)*pow(x0,2.0)" \
        "                             *pow(y0,4.0)" \
	"+3.0*pow(rx,2.0)*pow(ry,4.0)*pow(x0,4.0)" \
	"*pow(y0,2.0)+pow(ry,6.0)*pow(x0,6.0)," \
	"0.5))" \
	"/(pow(pow(rx,2.0)*pow(y0,2.0)+pow(ry,2.0)*pow(x0,2.0),1.0)" \
	"*pow(pow(rx,4.0)*pow(y0,6.0)+(pow(ry,4.0)+2.0*pow(rx,4.0))" \
	"*pow(x0,2.0)*pow(y0,4.0)" \
	"+(2.0*pow(ry,4.0)+pow(rx,4.0))" \
	"*pow(x0,4.0)*pow(y0,2.0)" \
	"+pow(ry,4.0)*pow(x0,6.0),0.5))"
    return format_expr(e)
}

function expr_egg(     ax, xp, yp, zp, rx, ry, ang, eg) {
    ax = $3 # XY, XZ, YZ
    xp=$5;  yp=$6;  zp=$7
    rx = $9; ry = $10
    ang = $12
    eg  = $14
    e[++m] =         "x  = xorg, y = yorg, z = zorg"
    if (prev_rot) {
	m = add_rc_cylinder(e, m) # like a cylinder
	m = add_rot(e, m)
    }
    e[++m] = sprintf("xp = %s, yp = %s, zp = %s", xp, yp, zp)
    e[++m] = sprintf("rx = %s, ry = %s", rx, ry)
    e[++m] = sprintf("ang = %s", ang)
    e[++m] = sprintf("eg  = %s", eg)    
    if (ax == "XY") {
	e[++m] = "x0 = sin(ang)*(y-yp)+cos(ang)*(x-xp)"
	e[++m] = "y0 = cos(ang)*(y-yp)-sin(ang)*(x-xp)"
    } else if (ax == "XZ") {
	e[++m] = "x0 = sin(ang)*(z-zp)+cos(ang)*(x-xp)"
	e[++m] = "y0 = cos(ang)*(z-zp)-sin(ang)*(x-xp)"
    } else if (ax == "YZ") {
	e[++m] = "x0 = sin(ang)*(z-zp)+cos(ang)*(y-yp)"
	e[++m] = "y0 = cos(ang)*(z-zp)-sin(ang)*(y-yp)"
    } else {
	printf "unknown ax in egg command (should be XY, XZ, YZ)\n" > "/dev/stderr"
	exit
    }
    e[++m] =			   \
"-(2.0*(pow(y0,2.0)+pow(x0,2.0)) " \
"     *(pow(pow(rx,2.0)*pow(y0,2.0)+pow(2.718281828459045, " \
"                                       (eg*rx*y0) " \
"                                        /(pow(ry,1.0) " \
"                                         *pow(pow(y0,2.0)+pow(x0,2.0),0.5))) " \
"                                   *pow(ry,2.0)*pow(x0,2.0),0.5) " \
"      -1.0*rx*ry) " \
"     *pow(pow(rx,6.0)*pow(y0,6.0)+3.0*pow(2.718281828459045, " \
"                                          (eg*rx*y0) " \
"                                           /(pow(ry,1.0) " \
"                                            *pow(pow(y0,2.0)+pow(x0,2.0), " \
"                                                 0.5)))*pow(rx,4.0) " \
"                                     *pow(ry,2.0)*pow(x0,2.0)*pow(y0,4.0) " \
"                                 +3.0*pow(2.718281828459045, " \
"                                          (2.0*eg*rx*y0) " \
"                                           /(pow(ry,1.0) " \
"                                            *pow(pow(y0,2.0)+pow(x0,2.0), " \
"                                                 0.5)))*pow(rx,2.0) " \
"                                     *pow(ry,4.0)*pow(x0,4.0)*pow(y0,2.0) " \
"                                 +pow(2.718281828459045, " \
"                                      (3.0*eg*rx*y0) " \
"                                       /(pow(ry,1.0) " \
"                                        *pow(pow(y0,2.0)+pow(x0,2.0),0.5))) " \
"                                  *pow(ry,6.0)*pow(x0,6.0),0.5)) " \
" /(pow(pow(rx,2.0)*pow(y0,2.0)+pow(2.718281828459045, " \
"                                   (eg*rx*y0) " \
"                                    /(pow(ry,1.0) " \
"                                     *pow(pow(y0,2.0)+pow(x0,2.0),0.5))) " \
"                               *pow(ry,2.0)*pow(x0,2.0),1.0) " \
"  *pow((-1.0*(4.0*pow(2.718281828459045, " \
"                      (2.0*eg*rx*y0)/(pow(ry,1.0) " \
"                                     *pow(pow(y0,2.0)+pow(x0,2.0),0.5)))*eg*rx " \
"                 *pow(ry,3.0)*pow(x0,4.0)*y0 " \
"             -4.0*pow(2.718281828459045, " \
"                      (eg*rx*y0)/(pow(ry,1.0) " \
"                                 *pow(pow(y0,2.0)+pow(x0,2.0),0.5)))*eg " \
"                 *pow(rx,3.0)*ry*pow(x0,4.0)*y0) " \
"            *pow(pow(y0,2.0)+pow(x0,2.0),0.5)) " \
"        +4.0*pow(rx,4.0)*pow(y0,6.0) " \
"        -1.0*pow(2.718281828459045, " \
"                 (2.0*eg*rx*y0)/(pow(ry,1.0) " \
"                                *pow(pow(y0,2.0)+pow(x0,2.0),0.5))) " \
"            *((-4.0*pow(ry,4.0)*pow(x0,2.0)*pow(y0,4.0)) " \
"             -8.0*pow(ry,4.0)*pow(x0,4.0)*pow(y0,2.0) " \
"             +((-4.0*pow(ry,4.0))-1.0*pow(eg,2.0)*pow(rx,2.0)*pow(ry,2.0)) " \
"              *pow(x0,6.0))+8.0*pow(rx,4.0)*pow(x0,2.0)*pow(y0,4.0) " \
"        +4.0*pow(rx,4.0)*pow(x0,4.0)*pow(y0,2.0),0.5)) "
    return format_expr(e)
}

function expr_sphere(m, xc, yc, zc, R, e) {
    xc = $3; yc=$4; zc=$5; R=$7
    e[++m] =         "x  = xorg, y = yorg, z = zorg"    
    e[++m] = sprintf("r2 = (x-%s)*(x-%s) + (y-%s)*(y-%s) + (z-%s)*(z-%s)",
		     xc, xc, yc, yc, zc, zc)
    e[++m] = sprintf("r0 = sqrt(r2)")
    e[++m] = sprintf("%s - r0", R)
    return format_expr(e)    
}

function upd_def(k, def, val) { # update if k == def
    return (k == def) ? val : k
}

function add_rc_block(e, m) { # add block center of rotation
    e[++m] = upd_def(xo, "xo",
		     sprintf("xo = 0.5*((%s) + (%s))", xlo, xhi))
    e[++m] = upd_def(yo, "yo",
		     sprintf("yo = 0.5*((%s) + (%s))", ylo, yhi))
    e[++m] = upd_def(zo, "zo",
		     sprintf("zo = 0.5*((%s) + (%s))", zlo, zhi))
    return m
}
function expr_block(     m, e) {
    xlo = $2; xhi=$3
    ylo = $4; yhi=$5
    zlo = $6; zhi=$7
    e[++m] =         "x  = xorg, y = yorg, z = zorg"
    if (prev_rot) {
	m = add_rc_block(e, m)
	m = add_rot(e, m)
    }
    e[++m] = sprintf("dX2 = sq(de(x, %s, %s)) + sq(di(y, %s, %s)) + sq(di(z, %s, %s))", \
		     xlo, xhi, ylo, yhi, zlo, zhi)
    e[++m] = sprintf("dY2 = sq(di(x, %s, %s)) + sq(de(y, %s, %s)) + sq(di(z, %s, %s))", \
		     xlo, xhi, ylo, yhi, zlo, zhi)
    e[++m] = sprintf("dZ2 = sq(di(x, %s, %s)) + sq(di(y, %s, %s)) + sq(de(z, %s, %s))", \
		     xlo, xhi, ylo, yhi, zlo, zhi)
    e[++m] = "dR2 = min3(dX2, dY2, dZ2)"
    e[++m] = "dR  = sqrt(dR2)"

    e[++m] = sprintf("in_box(x, y, z, %s, %s, %s, %s, %s, %s) ? dR : -dR", \
		     xlo, xhi, ylo, yhi, zlo, zhi)

    return format_expr(e)
}

function add_code_line(line, tab) {
    tab = gensp(8)
    csub("update_sdf", line "\n" tab  "//%update_sdf%")
}

function expr2code(expr) {
    add_code_line(format_line(expr))
}

# decide if we should invert the object
function set_invert(   fst) {
    fst = substr($1, 1, 1)
    INVERT = (fst == "!")
    if (INVERT) $1 = substr($1, 2) # cut a first charachter
}


function set_void_or_wall(  fst) {
    fst = substr($1, 1, 1)
    VOID_WINS = (fst == "|")
    if (VOID_WINS) $1 = substr($1, 2) # cut a first charachter
}

BEGIN {
    init()
    # read entire file
    while (getline < processor_file > 0) {
	processor = processor sep $0
	sep = ORS
    }
}

# strip suffix
function ss(s) {
    sub(/\.[^\.]*$/, "", s)
    return s
}

function basename1(s, arr, nn) {
    nn=split(s, arr, "/")
    return arr[nn]
}

function basename(s) {
    return ss(basename1(s))
}

function psystem(s) {
    printf "(tsdf) exec: %s\n", s > "/dev/stderr"
    system(s)
}

# uses variables CXX, CPPFLAGS, TD
function compile_and_run(f, args,      exec_name, c, r) {
    exec_name = TD "/" basename(f)
    c = sprintf("%s %s -o %s %s", CXX, CPPFLAGS, exec_name, f)
    psystem(c)
    r = sprintf("%s %s", exec_name, args)
    psystem(r)
}

END {
    if (EXIT_FAILURE) exit

    processor_code = sprintf("%s/processor.cpp", TD)
    printf "%s\n", processor > processor_code
    close(processor_code)

    compile_processor = sprintf("%s %s %s -o %s/processor", CXX, CPPFLAGS, processor_code, TD)
    wsystem(compile_processor)

    run_cmd     = sprintf("%s/processor %s", TD, sdf_file)
    wsystem(run_cmd)

    if (vtk_file) {
	sdf2vtk_cmd     = sprintf("sdf2vtk %s %s", sdf_file, vtk_file)
	wsystem(sdf2vtk_cmd)
    }
}

function parse_rot(   i) { # fills the center [xyz]r and angels
			   # phi[xyz] of rotation
    i = 1;
    i++; i++ # skip "rot center"
    xo   = $(i++); yo = $(i++); zo = $(i++) # center of rotation
    i++;     # skip "angles"
    phix = $(i++); phiy = $(i++); phiz = $(i++)
}

########### process config file ###########
{
    sub(/#.*/, "")         # strip comments
}

!NF {
    # skip empty lines
    next
}

$1=="extent" {
    xextent=$2; yextent=$3; zextent=$4
    psub("xextent", xextent)
    psub("yextent", yextent)
    psub("zextent", zextent)
    next
    
}

$1=="N" {
    NX = $2
    # if not given guess it from extents
    NY = NF < 3 ? yextent * (NX / xextent) : $3
    NZ = NF < 4 ? zextent * (NX / xextent) : $4
    psub("NX", NX); psub("NY", NY); psub("NZ", NZ);
    next
}

$1=="obj_margin" {
    obj_margin = $2
    psub("OBJ_MARGIN", obj_margin)
    next
}

$1 == "rot" {
    prev_rot = 1
    parse_rot()
    next
}

{
    set_invert()
    set_void_or_wall()
}


$1 == "plane" {
    expr2code(expr_plane())
}

$1 == "sphere" {
    expr2code(expr_sphere())
}

$1 == "cylinder" {
    expr2code(expr_cylinder())
}

$1 == "block" {
    expr2code(expr_block())
}

$1 == "ellipse" {
    expr2code(expr_ellipse())
}

$1 == "egg" {
    expr2code(expr_egg())
}

{
    prev_rot = 0 # previous command is not rotation
}
