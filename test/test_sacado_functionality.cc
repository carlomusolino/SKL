#include <iostream>

#include <grace_id/mappings/linear_mapping.hh>

#include <Sacado.hpp>



int main() {
    using namespace grace_id ; 
    // Check if Sacado is working by creating a simple FAD object
    Sacado::Fad::DFad<double> x = 3.0;
    x.diff(0, 1); // Set x as the independent variable
    linear_coordinate_mapping map {1,2} ; 
    Sacado::Fad::DFad<double> y = map(x); // Simple operation to test Sacado
    std::cout << "Value: " << y.val() << ", Derivative: " << y.dx(0) << std::endl;
    auto z = map.inverse(x) ;
    std::cout << "Inverse " << z.val() << ", Derivative " << z.dx(0) << std::endl ;
    return 0;
}
