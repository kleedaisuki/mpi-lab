set(MPILAB_PUBLIC_HEADERS
    "${PROJECT_SOURCE_DIR}/include/mpilab/application/mpi_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/application/pipeline.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/KernelLike.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/MatrixLike.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/kernel/AdvancedOneSidedJacobiSvd.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/kernel/MpiFriendlyOneSidedJacobiSvd.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/kernel.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/kernel/OneSidedJacobiSvd.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix/AdvancedMatrix.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix/DenseMatrix.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix/NaiveMatrix.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix/StridedMatrix.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/domain/matrix/detail/MatrixMemory.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/infrastructure/file_stream.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/infrastructure/logger.hpp"
    "${PROJECT_SOURCE_DIR}/include/mpilab/infrastructure/mpi_runtime.hpp"
)

set(MPILAB_SOURCES
    infrastructure/file_stream.cpp
    infrastructure/mpi_runtime.cpp
    infrastructure/logger.cpp
    application/pipeline.cpp
    application/mpi_scheduler.cpp
    domain/kernel/advanced_one_sided_jacobi_svd.cpp
    domain/kernel/detail/one_sided_jacobi_svd_detail.cpp
    domain/kernel/mpi_friendly_one_sided_jacobi_svd.cpp
    domain/kernel/one_sided_jacobi_svd.cpp
    domain/matrix/layout_utils.cpp
)

add_library(mpilab ${MPILAB_SOURCES} ${MPILAB_PUBLIC_HEADERS})
add_library(mpi_svd_lab::mpilab ALIAS mpilab)

target_include_directories(mpilab
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(mpilab PUBLIC cxx_std_20)

set_target_properties(mpilab
    PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME mpilab
        OUTPUT_NAME mpilab
)

mpilab_apply_common_target_options(mpilab)

if(MPILAB_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX QUIET)
    if(MPI_CXX_FOUND)
        target_link_libraries(mpilab PUBLIC MPI::MPI_CXX)
        target_compile_definitions(mpilab PUBLIC MPILAB_HAS_MPI=1)
        set(MPILAB_PACKAGE_USES_MPI TRUE PARENT_SCOPE)
    elseif(MPILAB_REQUIRE_MPI)
        message(FATAL_ERROR "MPILAB_REQUIRE_MPI is ON, but MPI CXX support was not found.")
    else()
        target_compile_definitions(mpilab PUBLIC MPILAB_HAS_MPI=0)
        message(STATUS "MPI CXX support was not found; building mpilab without MPI linkage.")
    endif()
endif()

install(
    TARGETS mpilab
    EXPORT mpi_svd_labTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp"
)
