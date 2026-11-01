#include "linear_solvers.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <algorithm>
#include <string>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// A: row-major, size n*n
bool dense_lu_solve(std::vector<double> A, std::vector<double> b, std::vector<double>& x, int n) {
    const double safe_eps = 1e-16;
    if ((int)b.size() != n) return false;
    x.assign(static_cast<size_t>(n), 0.0);

    std::vector<int> piv(n);
    for (int i = 0; i < n; ++i) piv[i] = i;

    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // find pivot in column k
        double maxval = 0.0;
        int pivrow = k;
        for (int i = k; i < n; ++i) {
            double val = std::abs(A[static_cast<size_t>(i * n + k)]);
            if (val > maxval) { maxval = val; pivrow = i; }
        }
        if (maxval < safe_eps) {
            // near-singular
            return false;
        }
        // swap rows k and pivrow in A and b
        if (pivrow != k) {
            for (int j = k; j < n; ++j) std::swap(A[static_cast<size_t>(k * n + j)], A[static_cast<size_t>(pivrow * n + j)]);
            std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(pivrow)]);
        }

        double Akk = A[static_cast<size_t>(k * n + k)];
        for (int i = k + 1; i < n; ++i) {
            double factor = A[static_cast<size_t>(i * n + k)] / Akk;
            A[static_cast<size_t>(i * n + k)] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                A[static_cast<size_t>(i * n + j)] -= factor * A[static_cast<size_t>(k * n + j)];
            }
            b[static_cast<size_t>(i)] -= factor * b[static_cast<size_t>(k)];
        }
    }

    // back substitution
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) s -= A[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
        double diag = A[static_cast<size_t>(i * n + i)];
        if (std::abs(diag) < safe_eps) return false;
        x[static_cast<size_t>(i)] = s / diag;
    }

    return true;
}

bool banded_lu_solve(std::vector<double> band, int n, int kl, int ku,
                     std::vector<double> b, std::vector<double>& x) {
    const double safe_eps = 1e-300;
    if (static_cast<int>(b.size()) != n) return false;
    const int w = band_storage_width(kl, ku);
    const int kuf = ku + kl; // upper bandwidth including fill-in
    auto at = [&](int r, int c) -> double& {
        return band[static_cast<size_t>(r) * static_cast<size_t>(w) + static_cast<size_t>(c - r + kl)];
    };

    for (int k = 0; k < n; ++k) {
        const int last_row = std::min(n - 1, k + kl);
        int pivrow = k;
        double maxval = std::abs(at(k, k));
        for (int i = k + 1; i <= last_row; ++i) {
            const double val = std::abs(at(i, k));
            if (val > maxval) { maxval = val; pivrow = i; }
        }
        if (maxval < safe_eps) return false;

        const int last_col = std::min(n - 1, k + kuf);
        if (pivrow != k) {
            // Row pivrow has no entries beyond pivrow + ku <= k + kuf, so the swap stays in band.
            for (int j = k; j <= last_col; ++j) std::swap(at(k, j), at(pivrow, j));
            std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(pivrow)]);
        }

        const double akk = at(k, k);
        for (int i = k + 1; i <= last_row; ++i) {
            const double factor = at(i, k) / akk;
            if (factor == 0.0) continue;
            at(i, k) = 0.0;
            for (int j = k + 1; j <= last_col; ++j) at(i, j) -= factor * at(k, j);
            b[static_cast<size_t>(i)] -= factor * b[static_cast<size_t>(k)];
        }
    }

    x.assign(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<size_t>(i)];
        const int last_col = std::min(n - 1, i + kuf);
        for (int j = i + 1; j <= last_col; ++j) s -= at(i, j) * x[static_cast<size_t>(j)];
        x[static_cast<size_t>(i)] = s / at(i, i);
    }
    return true;
}

namespace {
// Resident and peak resident memory of this process in MB (Linux /proc), -1 if unavailable.
void process_memory_mb(double& rss, double& hwm) {
    rss = hwm = -1.0;
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        long kb = 0;
        if (std::sscanf(line, "VmRSS: %ld kB", &kb) == 1) rss = kb / 1024.0;
        if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) hwm = kb / 1024.0;
    }
    std::fclose(f);
}
}

struct SparseLUSolver::Impl {
    using SpD = Eigen::SparseMatrix<double>;
    using SpF = Eigen::SparseMatrix<float>;
    std::unique_ptr<Eigen::SparseLU<SpD, Eigen::COLAMDOrdering<int>>> lu;
    std::unique_ptr<Eigen::SparseLU<SpF, Eigen::COLAMDOrdering<int>>> luf;
    std::unique_ptr<Eigen::IncompleteLUT<double>> ilu;
    Kind kind = Kind::LU;
    double pivot = 1.0, fill = 20.0, droptol = 1e-5;
    std::string err;
    int n = 0;
};

SparseLUSolver::SparseLUSolver() : impl_(new Impl) {}
SparseLUSolver::~SparseLUSolver() { delete impl_; }

bool SparseLUSolver::factor(int n, const std::vector<Entry>& entries, Kind kind) {
    // Release the factorizations of the other kinds before building this one (the kept one of
    // the same kind reuses its storage).
    if (kind != Kind::LU) impl_->lu.reset();
    if (kind != Kind::LUFloat) impl_->luf.reset();
    if (kind != Kind::ILUT) impl_->ilu.reset();
    impl_->kind = kind;
    Eigen::SparseMatrix<double> A(n, n);
    {
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(entries.size());
        for (const Entry& e : entries) trip.emplace_back(e.row, e.col, e.val);
        A.setFromTriplets(trip.begin(), trip.end());
    }
    A.makeCompressed();
    impl_->n = n;
    // Diagnostics (environment): BS_DUMP_JACOBIAN=file writes the matrix (int64 n, int64 nnz, then
    // nnz x {int row, int col, double val}; overwritten at every factorization, so the file holds
    // the last one, unless the name contains a printf integer format such as %d or %03d, which is
    // replaced by the factorization count);
    // BS_LU_STATS=1 prints the fill and the process memory after each factorization.
    static int n_factor = 0;
    ++n_factor;
    if (const char* dump = std::getenv("BS_DUMP_JACOBIAN")) {
        char name[1024];
        if (std::strchr(dump, '%')) std::snprintf(name, sizeof name, dump, n_factor);
        else std::snprintf(name, sizeof name, "%s", dump);
        std::ofstream out(name, std::ios::binary);
        const long long nn = n, ne = static_cast<long long>(entries.size());
        out.write(reinterpret_cast<const char*>(&nn), sizeof nn);
        out.write(reinterpret_cast<const char*>(&ne), sizeof ne);
        out.write(reinterpret_cast<const char*>(entries.data()), static_cast<std::streamsize>(ne * sizeof(Entry)));
    }
    bool ok = false;
    long nnz_factor = 0;
    const char* name = "LU";
    if (kind == Kind::LU) {
        if (!impl_->lu) impl_->lu.reset(new Eigen::SparseLU<Impl::SpD, Eigen::COLAMDOrdering<int>>);
        impl_->lu->setPivotThreshold(impl_->pivot);
        impl_->lu->analyzePattern(A);
        impl_->lu->factorize(A);
        ok = impl_->lu->info() == Eigen::Success;
        if (!ok) impl_->err = impl_->lu->lastErrorMessage();
        else nnz_factor = static_cast<long>(impl_->lu->nnzL() + impl_->lu->nnzU());
    } else if (kind == Kind::LUFloat) {
        name = "float LU";
        if (!impl_->luf) impl_->luf.reset(new Eigen::SparseLU<Impl::SpF, Eigen::COLAMDOrdering<int>>);
        impl_->luf->setPivotThreshold(static_cast<float>(impl_->pivot));
        const Impl::SpF Af = A.cast<float>();
        A.resize(0, 0);
        A.data().squeeze();
        impl_->luf->analyzePattern(Af);
        impl_->luf->factorize(Af);
        ok = impl_->luf->info() == Eigen::Success;
        if (!ok) impl_->err = impl_->luf->lastErrorMessage();
        else nnz_factor = static_cast<long>(impl_->luf->nnzL() + impl_->luf->nnzU());
    } else {
        name = "ILUT";
        if (!impl_->ilu) impl_->ilu.reset(new Eigen::IncompleteLUT<double>);
        impl_->ilu->setFillfactor(static_cast<int>(impl_->fill));
        impl_->ilu->setDroptol(impl_->droptol);
        impl_->ilu->compute(A);
        ok = impl_->ilu->info() == Eigen::Success;
        if (!ok) impl_->err = "incomplete LU breakdown (zero pivot)";
    }
    if (std::getenv("BS_LU_STATS")) {
        double rss, hwm;
        process_memory_mb(rss, hwm);
        std::printf("[lu] #%d %s n = %d, nnz(L+U) = %ld%s, RSS %.0f MB, peak RSS %.0f MB\n", n_factor, name, n,
                    nnz_factor, (kind == Kind::ILUT) ? " (not reported for ILUT)" : "", rss, hwm);
        std::fflush(stdout);
    }
    return ok;
}

bool SparseLUSolver::solve(const std::vector<double>& b, std::vector<double>& x) const {
    Eigen::Map<const Eigen::VectorXd> bv(b.data(), impl_->n);
    if (impl_->kind == Kind::LU && impl_->lu) {
        Eigen::VectorXd xv = impl_->lu->solve(bv);
        if (impl_->lu->info() != Eigen::Success) return false;
        x.assign(xv.data(), xv.data() + impl_->n);
        return true;
    }
    if (impl_->kind == Kind::LUFloat && impl_->luf) {
        const Eigen::VectorXf bf = bv.cast<float>();
        const Eigen::VectorXf xf = impl_->luf->solve(bf);
        if (impl_->luf->info() != Eigen::Success) return false;
        x.resize(static_cast<size_t>(impl_->n));
        for (int k = 0; k < impl_->n; ++k) x[static_cast<size_t>(k)] = static_cast<double>(xf(k));
        return true;
    }
    if (impl_->kind == Kind::ILUT && impl_->ilu) {
        Eigen::VectorXd xv = impl_->ilu->solve(bv);
        x.assign(xv.data(), xv.data() + impl_->n);
        return true;
    }
    return false;
}

SparseLUSolver::Kind SparseLUSolver::kind() const { return impl_->kind; }

const char* SparseLUSolver::last_error() const { return impl_->err.c_str(); }

void SparseLUSolver::set_pivot_threshold(double t) { impl_->pivot = t; }

void SparseLUSolver::set_ilut(double fill, double droptol) {
    impl_->fill = fill;
    impl_->droptol = droptol;
}
