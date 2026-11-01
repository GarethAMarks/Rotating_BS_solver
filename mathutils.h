#ifndef MATHUTILS_HPP_
#define MATHUTILS_HPP_

#include<string.h>
#include<sstream>
#include<vector>
#include<cmath>
//#include"mathutils.cpp"

using std::string;

//just a header file for miscellaneous functions to avoid cluttering class-specific files
double cubic_interp(double r, double f0, double f1, double f2, double f3, int j0, double dr);
int bound(int n, int lower, int upper);
// 4th-order accurate uniform-grid interpolation (wrapper around cubic_interp)
// f: array of nodal values at x = k * dr for k=0..N-1
// x: target evaluation point (in same coordinate units)
// dr: grid spacing of the f array
// Returns cubic (4-point Lagrange) interpolation at x. Falls back to linear/quadratic when near boundaries.
double interp4_uniform(const std::vector<double>& f, double x, double dr);

// Variant that accepts an explicit starting index j0 (index of f[0] used by cubic_interp)
// j0 must satisfy 0 <= j0 <= N-4. The function will clamp j0 into range if needed.
double interp4_uniform_with_j0(const std::vector<double>& f, double x, int j0, double dr);

// 5-point (degree-4) Lagrange interpolation for uniform grids
// In smooth regions this yields O(dr^5) error; using it ensures interpolation won't be the accuracy bottleneck
// when combined with 4th-order spatial derivatives. Near boundaries, callers may prefer to fall back to
// interp4_uniform or a one-sided stencil.
double interp5_uniform_with_j0(const std::vector<double>& f, double x, int j0, double dr);
double interp5_uniform(const std::vector<double>& f, double x, double dr);

// Even-symmetric interpolation near the origin (assumes f(-x) = f(x))
// These versions reflect indices across 0 when a left-boundary stencil would need negative indices.
double interp4_uniform_even(const std::vector<double>& f, double x, double dr);
double interp5_uniform_even(const std::vector<double>& f, double x, double dr);
double fivePointDeriv(double step, int order, double f1, double f2, double f3, double f4, double f5);
double sevenPointDeriv(double step, int order, double f1, double f2, double f3, double f4, double f5, double f6, double f7);
void computeSplineCoefficients(const std::vector<double>& y, std::vector<double>& M);
double evaluateSpline(const std::vector<double>& y, const std::vector<double>& secondDerivatives, int i, double x);
double evaluateSplineSegment(const std::vector<double>& x, const std::vector<double>& y,
                              const std::vector<double>& M, int seg, double xi);
double lagrange_interp(double x, const std::vector<double>& X, const std::vector<double>& Y);

//Below are templated, so we put these in the header directly

//stencil for a derivative evaluated about the 4th position on uniform grid
template <typename T>
T p4_stencil(double step, T f1, T f2, T f3, T f4, T f5)
{
    return (-(1.0 / 12.0) * f1 + (1.0 / 2.0) * f2 - (3.0 / 2.0) * f3 + (5.0 / 6.0) * f4 + (1.0 / 4.0) * f5) / step;
}
template <typename T>
T p5_stencil(double step, T f1, T f2, T f3, T f4, T f5)
{
    return ((1.0 / 4.0) * f1 - (4.0 / 3.0) * f2 + (3.0) * f3 - (4.0) * f4 + (25.0 / 12.0) * f5) / step;
}

template <typename T>
T sevenPointDeriv(double step, int order, T f1, T f2, T f3, T f4, T f5, T f6, T f7)
{
    switch (order)
    {
        case 0:
            return f4;
        case 1:
            return ((-1.0/60.0)*f1 + (3.0/20.0)*f2 - (3.0/4.0)*f3 + (3.0/4.0)*f5 -(3.0/20.0)*f6 + (1.0/60.0)*f7 )/step;
        case 2:
            return ((1.0/90.0)*f1 -(3.0/20.0)*f2 + (3.0/2.0)*f3 - (49.0/18.0)*f4 + (3.0/2.0)*f5 -(3.0/20.0)*f6 + (1.0/90.0)*f7)/pow(step,2);
        case 3:
            return ((1.0/8.0)*f1 -(1)*f2 + (13.0/8.0)*f3 - (13.0/8.0)*f5 +(1)*f6 - (1.0/8.0)*f7)/pow(step,3);
        case 4:
            return (-(1.0/6.0)*f1 +(2.0)*f2 - (13.0/2.0)*f3 + (28.0/3.0)*f4 - (13.0/2.0)*f5 +(2.0)*f6 - (1.0/6.0)*f7)/pow(step,4);
        case 5:
            return ((-1.0/2.0)*f1 +(2.0)*f2 - (5.0/2.0)*f3  + (5.0/2.0)*f5 -(2.0)*f6 + (1.0/2.0)*f7)/pow(step,5);
        case 6:
            return ((1.0)*f1 -(6.0)*f2 + (15.0)*f3 - (20.0)*f4 + (15.0)*f5 -(6.0)*f6 + (1.0)*f7)/pow(step,6);
        default:
            printf("ERROR: invalid derivative order requested in sevenPointDeriv");
            abort();
    }
}

//helper function for reading parameters from BSParams.par. Might need to add smth for array types
template <typename T>
void fill_parameter (string& current_line, string line_start, T& parameter, bool quiet)
{
    // Strip inline comments starting with '#'
    size_t hash = current_line.find('#');
    if (hash != string::npos) current_line = current_line.substr(0, hash);

    // Require exact prefix match at start of (trimmed) line to avoid partial matches
    size_t pos = current_line.find_first_not_of(" \t");
    if (pos == string::npos) return;
    if (current_line.compare(pos, line_start.length(), line_start) != 0) return;

    // Substring starting immediately after the prefix
    string rest_of_line = current_line.substr(pos + line_start.length());

    std::stringstream ss(rest_of_line);
    T value;
    if constexpr (std::is_same<T, std::string>::value) {
        // strings (paths) may contain spaces: take the rest of the line, trimmed
        const size_t a = rest_of_line.find_first_not_of(" \t\r");
        const size_t b = rest_of_line.find_last_not_of(" \t\r");
        if (a != string::npos) {
            parameter = rest_of_line.substr(a, b - a + 1);
            if (!quiet) std::cout << "Read in " << line_start << parameter << std::endl;
        } else {
            std::cout << "WARNING: Failed to extract value for parameter " << line_start << std::endl;
        }
        return;
    }
    if (ss >> value) {
        parameter = value;
        if (!quiet) std::cout << "Read in " << line_start << parameter << std::endl;
    } else {
        std::cout << "WARNING: Failed to extract value for parameter " << line_start << std::endl;
    }
}

//similar to above but reads in arrays of numeric types of arbitrary type, separated by spaces
template <typename T>
void fill_param_array (string& current_line, string line_start, std::vector<T>& param_array, bool quiet)
{
    // Strip inline comments starting with '#'
    size_t hash = current_line.find('#');
    if (hash != string::npos) current_line = current_line.substr(0, hash);

    // Require exact prefix match at start of (trimmed) line to avoid partial matches
    size_t pos = current_line.find_first_not_of(" \t");
    if (pos == string::npos) return;
    if (current_line.compare(pos, line_start.length(), line_start) != 0) return;

    string rest_of_line = current_line.substr(pos + line_start.length());

    std::stringstream ss(rest_of_line);
    T value;
    while (ss >> value)
    {
        param_array.push_back(value);
    }
    std::cout << "Read in " << line_start;

    unsigned int k = 0;
    while (!quiet && k < param_array.size() )
    {
        std::cout  << param_array[k] <<  ", ";
        k++;
    }
    std::cout << std::endl;

    if (param_array.size() == 0) std::cout << "WARNING: Failed to extract value for parameter array" << line_start << std::endl;
}
#endif /* MATHUTILS_HPP_ */
