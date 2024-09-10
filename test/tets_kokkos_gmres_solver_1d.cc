#include <Sacado.hpp>
#include <vector>
#include <iostream>

#include <Kokkos_Core.hpp> 

int main () {

    constexpr size_t m = 10 ; // vector size 
    constexpr size_t p = 1   ; // derivative dimension 

    using fad_t = Sacado::Fad::SFad<double, p> ;  

    Kokkos::initialize() ;
    {
        Kokkos::View<fad_t*> u("U", m, p+1)  ;
        Kokkos::View<fad_t*> res("res", m, p+1) ; 
        Kokkos::View<fad_t*> v("v", m, p); 

        double xm{-1}, xp{1}, h{(xp-xm)/(m-1)} ; 
        Kokkos::parallel_for( "set_values", m,
            KOKKOS_LAMBDA ( int i ) {
                u(i) = Kokkos::sin(M_PI * (xm + i * h) ) ; 
                v(i) = 1./m ; 
                u(i).fastAccessDx(0) = v(i).val() ; 
            }
        ); 

        Kokkos::parallel_for( "get_residual", m,
            KOKKOS_LAMBDA ( int i ) {
                res(i) = u(i) * u(i) - 2. * u(i) ; 
            }
        ); 

        auto h_res = Kokkos::create_mirror_view(res) ; 
        Kokkos::deep_copy(h_res, res) ; 
        for( int ii=0; ii<10; ++ii) {
            std::cout << "f(u): " << h_res(ii).val() << " deriv: " << h_res(ii).fastAccessDx(0) << std::endl ; 
        }
    }
    Kokkos::finalize() ;
    return EXIT_SUCCESS ; 
}