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

// (row, col, value) entry of a sparse matrix; duplicates are summed when a matrix is built.
struct SparseEntry { int row, col; double val; };

// Square sparse matrix in compressed column storage (Eigen::SparseMatrix<double> behind the
// interface). Built once per Jacobian and shared by the factorization and the products, so
// that no other copy of the matrix is kept.
class SparseMatrix {
public:
    SparseMatrix();
    ~SparseMatrix();
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;

    // Build an n x n matrix from lists of entries (any order; duplicates summed). Each list is
    // released as soon as it has been copied, so the entries and the matrix coexist only briefly.
    void build(int n, std::vector<std::vector<SparseEntry>>& parts);
    void clear();                                     // release the storage
    int rows() const;
    long nnz() const;
    // y = A x
    void multiply(const std::vector<double>& x, std::vector<double>& y) const;
    // Every stored entry (diagnostics: dense comparisons, dumps).
    void entries(std::vector<SparseEntry>& out) const;

private:
    struct Impl;
    Impl* impl_;
    friend class SparseLUSolver;
};

// Sparse factorization behind a small interface so that the solver code does not include Eigen;
// factor once, then solve (or precondition) for several right-hand sides. Kinds:
//   LU       exact sparse LU in double precision (Eigen::SparseLU, COLAMD ordering): a direct solver;
//   LUFloat  the same in single precision (about half the memory): a preconditioner, since a
//            solve is only accurate to ~1e-7 x condition number;
//   ILUT     incomplete LU with threshold dropping (Eigen::IncompleteLUT): a preconditioner whose
//            size grows roughly linearly with the grid instead of with the LU fill; its factors
//            can be kept in single precision (set_ilut(..., single = true)).
class SparseLUSolver {
public:
    using Entry = SparseEntry;
    enum class Kind { LU, LUFloat, ILUT };
    SparseLUSolver();
    ~SparseLUSolver();
    SparseLUSolver(const SparseLUSolver&) = delete;
    SparseLUSolver& operator=(const SparseLUSolver&) = delete;

    // Factor A (not copied for the LU and ILUT kinds; the float LU works on a single-precision
    // copy). Returns false if the factorization fails (for ILUT: a zero pivot). The factorization
    // held before is released first, so the old and new factors never coexist.
    bool factor(const SparseMatrix& A, Kind kind = Kind::LU);
    // The same from entries (duplicate (row, col) entries are summed).
    bool factor(int n, const std::vector<Entry>& entries, Kind kind = Kind::LU);
    // Partial-pivoting threshold in (0, 1] for the LU kinds: < 1 prefers the diagonal when it is
    // within this factor of the column maximum, which reduces fill (SuperLU's diag_pivot_thresh).
    void set_pivot_threshold(double t);
    // ILUT: keep at most fill x (nonzeros of the row of A) entries per row of L and of U, and
    // drop entries below droptol x (row norm); single: store the factors in float (2/3 of the
    // memory per entry; float round-off grows with the condition number, so on ill-conditioned
    // Jacobians it can dominate the dropping and weaken the preconditioner).
    void set_ilut(double fill, double droptol, bool single = false);
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
