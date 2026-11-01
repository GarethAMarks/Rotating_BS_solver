#ifndef LINEAR_SOLVERS_H_
#define LINEAR_SOLVERS_H_

#include <vector>

// Solve dense linear system A x = b where A is stored row-major (n*n entries).
// Returns true on success; x is resized to n. This uses Gaussian elimination with
// partial pivoting. On failure (e.g. singular), returns false.
bool dense_lu_solve(std::vector<double> A, std::vector<double> b, std::vector<double>& x, int n);

// Row width of the band storage used by banded_lu_solve.
inline int band_storage_width(int kl, int ku) { return 2 * kl + ku + 1; }

// Solve A x = b for a band matrix with kl sub- and ku super-diagonals by Gaussian
// elimination with partial pivoting. Storage is row-major:
//   A(r, c) = band[r * w + (c - r + kl)],  w = band_storage_width(kl, ku),
// valid for -kl <= c - r <= ku + kl. The extra kl super-diagonals hold pivoting
// fill-in and must be zero on entry. Returns false if a zero pivot is met.
bool banded_lu_solve(std::vector<double> band, int n, int kl, int ku,
                     std::vector<double> b, std::vector<double>& x);

// Sparse factorization behind a small interface so that the solver code does not include Eigen;
// factor once, then solve (or precondition) for several right-hand sides. Kinds:
//   LU       exact sparse LU in double precision (Eigen::SparseLU, COLAMD ordering): a direct solver;
//   LUFloat  the same in single precision (about half the memory): a preconditioner, since a
//            solve is only accurate to ~1e-7 x condition number;
//   ILUT     incomplete LU with threshold dropping (Eigen::IncompleteLUT): a preconditioner whose
//            size grows roughly linearly with the grid instead of with the LU fill.
class SparseLUSolver {
public:
    struct Entry { int row, col; double val; };
    enum class Kind { LU, LUFloat, ILUT };
    SparseLUSolver();
    ~SparseLUSolver();
    SparseLUSolver(const SparseLUSolver&) = delete;
    SparseLUSolver& operator=(const SparseLUSolver&) = delete;

    // Duplicate (row, col) entries are summed. Returns false if the factorization fails (for
    // ILUT: a zero pivot). The factorization of any other kind held before is released first.
    bool factor(int n, const std::vector<Entry>& entries, Kind kind = Kind::LU);
    // Partial-pivoting threshold in (0, 1] for the LU kinds: < 1 prefers the diagonal when it is
    // within this factor of the column maximum, which reduces fill (SuperLU's diag_pivot_thresh).
    void set_pivot_threshold(double t);
    // ILUT: keep at most fill x (nonzeros of the row of A) entries per row of L and of U, and
    // drop entries below droptol x (row norm).
    void set_ilut(double fill, double droptol);
    // Apply the factorization: x = M^{-1} b (exact solve for Kind::LU).
    bool solve(const std::vector<double>& b, std::vector<double>& x) const;
    Kind kind() const;
    bool exact() const { return kind() == Kind::LU; }
    const char* last_error() const;

private:
    struct Impl;
    Impl* impl_;
};

#endif // LINEAR_SOLVERS_H_
