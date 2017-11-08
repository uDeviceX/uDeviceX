Adapted from [1]

# abstract
We report on an accurate and efficient algorithm for obtaining all
roots of general real cubic and quartic polynomials. Both the cubic
and quartic solvers give highly accurate roots and place no
restrictions on the magnitude of the polynomial coefficients. The key
to the algorithm is a proper rescaling of both polynomials. This puts
upper bounds on the magnitude of the roots and is very useful in
stabilizing the root finding process. The cubic solver is based on
dividing the cubic polynomial into six classes. By analyzing the root
surface for each class, a fast convergent Newton-Raphson starting
point for a real root is obtained at a cost no higher than three
additions and four multiplications. The quartic solver uses the cubic
solver in getting information about stationary points and, when the
quartic has real roots, stable Newton-Raphson iterations give one of
the extreme real roots. The remaining roots follow by composite
deflation to a cubic. If the quartic has only complex roots, the
present article shows that a stable Newton-Raphson iteration on a
derived symmetric sixth degree polynomial can be formulated for the
real parts of the complex roots. The imaginary parts follow by solving
suitable quadratics.

# refs
[1] Flocke, N. (2015). Algorithm 954: An Accurate and Efficient Cubic
and Quartic Equation Solver for Physical Applications. ACM
Transactions on Mathematical Software (TOMS), 41(4), 30.
