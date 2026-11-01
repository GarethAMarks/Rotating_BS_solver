#include <iostream>
#include <string>

#include "params_io.h"
#include "solver.h"
#include "rotating_solver.h"

int main(int argc, char* argv[]) {
    std::string param_file = "rotating_params.par";
    if (argc > 1) {
        param_file = argv[1];
    }

    try {
        BSParams p = read_params(param_file);
        print_params_summary(p);

        if (p.sequence_mode == "rotating") {
            RotatingSolver rsolver(p);
            const bool ok = rsolver.run();
            std::cout << (ok ? "[ok] Rotating solve completed.\n" : "[warn] Rotating solve did not complete.\n");
            return ok ? 0 : 1;
        }

        BosonStarSolver solver(p);
        solver.initialize_seed();

        const bool converged = solver.run_relaxation_skeleton();
        solver.print_run_summary();

        if (!converged) {
            std::cerr << "[warn] Residual solver did not reach requested tolerance within max iterations.\n";
            std::cerr << "[warn] Continuing anyway (prototype stage).\n";
            return 0;
        }

        std::cout << "[ok] Residual solve completed for spherical system.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
