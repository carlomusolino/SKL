#include <iostream>

#include <grace_id_config.h>

#include <grace_id/mappings/linear_mapping.hh>

#include <Sacado.hpp>
#include <Kokkos_Core.hpp>

#define N_POINTS 15

using dfad_t = Sacado::Fad::DFad<double>;

// Function to compute N Chebyshev-Gauss-Lobatto collocation points in [-1, 1]
std::vector<dfad_t> chebyshev_points(int N) {
    std::vector<dfad_t> points(N);
    for (int i = 0; i < N; ++i) {
        points[i] = -std::cos(M_PI * i / (N - 1));  // Chebyshev-Gauss-Lobatto points formula
    }
    return points;
}

// Function to compute Chebyshev coefficients from function values
std::vector<dfad_t> chebyshev_coefficients(const std::vector<dfad_t>& f) {
    int N = f.size();
    std::vector<dfad_t> c(N, 0.0);
    for (int k = 0; k < N; ++k) {
        dfad_t sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += f[j] * cos(M_PI * k * j / (N - 1));
        }
        if (k == 0 || k == N - 1) {
            c[k] = sum / static_cast<double>(N - 1);
        } else {
            c[k] = 2. * sum / static_cast<double>(N - 1);
        }
    }
    return c;
}

// Function to reconstruct function values from Chebyshev coefficients
std::vector<dfad_t> chebyshev_reconstruct(const std::vector<dfad_t>& c) {
    int N = c.size();
    std::vector<dfad_t> f(N, 0.0);
    for (int j = 0; j < N; ++j) {
        dfad_t sum = 0; // Start with the first term, divided by 2
        for (int k = 0; k < N ; ++k) {
            sum += c[k] * cos(M_PI * k * j / (N - 1));
        }
        //sum += 0.5 * c[N - 1] * cos(M_PI * (N - 1) * j / (N - 1)); // Last term, divided by 2
        f[j] = sum;
    }
    return f;
}

std::vector<dfad_t> chebyshev_deriv( const std::vector<dfad_t>& c ) {
    int N = c.size() ; 
    std::vector<dfad_t> c_d(N, 0.0);
    c_d[N-2] = 2*(N-1)*c[N-1] ; 
    for(int k=N-3; k>0; --k) {
        c_d[k] = c_d[k+2] + 2*(k+1)*c[k+1] ; 
    }
    c_d[0] = 0.5 * c_d[2] + c[1] ;  
    return c_d ; 
}

std::vector<dfad_t> chebyshev_second_derivative(const std::vector<dfad_t>& c) {
    auto const c_d = chebyshev_deriv(c) ;
    return chebyshev_deriv(c_d);
}
#if 1
std::vector<dfad_t> residual( std::vector<dfad_t> const& u, std::vector<dfad_t> const& x, std::vector<double> const& bc ) {
    auto const N = u.size() ; 
    std::vector<dfad_t> s(N) ; 
    for( int i=0; i<N; ++i) {
        s[i] = M_PI*M_PI * std::sin(M_PI*x[i]) ; 
    }
    auto const utilde = chebyshev_coefficients(u) ; 
    auto const utilde2 = chebyshev_second_derivative(utilde) ; 
    auto const d2udx2 = chebyshev_reconstruct(utilde2) ; 

    std::vector<dfad_t> residual( N, 0. ) ; 
    for( int i=1; i<N-1; ++i) { 
        residual[i] = -d2udx2[i] - s[i] ; 
    }
    residual[0] = u[0] - bc[0] ; 
    residual[N-1] = u[N-1] - bc[1] ; 
    return residual ; 
}
#endif 
int main() {
    int N = 10;  // Example number of collocation points
    double u0{0}, u1{0}  ; //Boundary conditions 
    // Example: Create function values at Chebyshev points using Sacado
    auto points = chebyshev_points(N);
    std::vector<dfad_t> u(N);
    for (int i = 0; i < N; ++i) {
        //exact_solution[i] = std::sin(M_PI * points[i]);  // Example function values
        u[i] = std::sin(M_PI * points[i]) ; 
    }

    // Compute Chebyshev coefficients
    auto coeffs = chebyshev_coefficients(u);
    for( int i=0; i< N; ++i) {
        coeffs[i].diff(i,N) ; 
    }
    //auto u = chebyshev_reconstruct(coeffs) ; 

    #if 0 
    auto coeffs_dudx2 = chebyshev_second_derivative(coeffs) ; 

    // Reconstruct function values from coefficients
    auto reconstructed_values = chebyshev_reconstruct(coeffs);

    auto reconstructed_deriv = chebyshev_reconstruct(coeffs_dudx2);

    // Output the reconstructed values and derivatives
    for (int i = 0; i < N; ++i) {
        std::cout << "Value: " << reconstructed_values[i].val()
                 << ", Second Derivative: " << reconstructed_deriv[i].val() << std::endl;
        std::cout << "Exact: " << u[i].val()  ; 
        std::cout << " Deriv Exact: " << -M_PI*M_PI*std::sin(M_PI*points[i].val()) << std::endl ; 
        std::cout << std::endl ;
    }
    #endif 


    return 0;
}