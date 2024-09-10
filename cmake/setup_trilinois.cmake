if(NOT TRILINOIS_ROOT)
    set(TRILINOIS_ROOT "")
    set(TRILINOIS_ROOT "$ENV{TRILINOIS_ROOT}")
endif()

message(STATUS "Searching path ${TRILINOIS_ROOT}")

# Find Trilinos package
find_package(Trilinos REQUIRED PATHS "${TRILINOIS_ROOT}")

message(STATUS "Trilinos libraries: ${Trilinos_LIBRARIES}")
message(STATUS "Trilinos includes: ${Trilinos_INCLUDE_DIRS}")

# Check if Sacado is enabled
if (Trilinos_FOUND AND Trilinos_ENABLE_Sacado)
    message(STATUS "Sacado functionality is present in Trilinos.")
else()
    message(WARNING "Sacado is not found in Trilinos or is not enabled.")
endif()

# Create imported target if not already defined
if(NOT TARGET Trilinos::Trilinos)
    add_library(Trilinos::Trilinos IMPORTED INTERFACE)
    set_property(TARGET Trilinos::Trilinos APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${Trilinos_INCLUDE_DIRS}")
    set_property(TARGET Trilinos::Trilinos APPEND PROPERTY
                 INTERFACE_LINK_LIBRARIES "${Trilinos_LIBRARIES}")
endif()
