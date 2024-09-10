#include <Sacado.hpp>
#include <vector>
#include <iostream>

using FadType = Sacado::Fad::DFad<double>;

// Example residual function
std::vector<FadType> compute_residual(const std::vector<FadType>& u) {
    std::vector<FadType> res(u.size());
    for (size_t i = 0; i < u.size(); ++i) {
        res[i] = u[0] * u[i] - 2.0 * u[i]; // Example: f(u) = u^2 - 2u
    }
    return res;
}

// Compute JVP using AD
std::vector<double> compute_jvp(const std::vector<double>& u_val, const std::vector<double>& v) {
    size_t N = u_val.size();
    std::vector<FadType> u(N);

    // Initialize FadType with directional derivatives
    for (size_t i = 0; i < N; ++i) {
        u[i] = FadType(1, u_val[i]);  // Initialize with value and 1 derivative
        u[i].fastAccessDx(0) = v[i];  // Set direction as perturbation vector v
    }

    // Compute residual using AD
    auto res = compute_residual(u);

    // Extract JVP from derivatives
    std::vector<double> jvp(N);
    for (size_t i = 0; i < N; ++i) {
        jvp[i] = res[i].dx(0);  // Get J*v
    }

    return jvp;
}

int main() {
    // Example input values
    std::vector<double> u_val = {1.0, 2.0, 3.0}; // Solution vector
    std::vector<double> v = {0.1, 0.2, 0.3};     // Perturbation vector

    // Compute JVP
    auto jvp = compute_jvp(u_val, v);

    // Output JVP result
    std::cout << "Jv: ";
    for (double val : jvp) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
