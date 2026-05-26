cmake_minimum_required(VERSION 3.20...3.29)

project(mpi_svd_lab
    VERSION 0.1.0
    DESCRIPTION "MPI SVD laboratory project"
    HOMEPAGE_URL "https://github.com/kleedaisuki/mpi-lab"
    LANGUAGES CXX
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(MpiLabOptions)

mpilab_setup_project_options()

set(MPILAB_PACKAGE_USES_MPI FALSE)

add_subdirectory(src/mpilab)

if(MPILAB_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mpi_svd_labConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/mpi_svd_labConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/mpi_svd_labConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mpi_svd_lab"
)

install(
    EXPORT mpi_svd_labTargets
    NAMESPACE mpi_svd_lab::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mpi_svd_lab"
)

install(
    FILES
        "${PROJECT_BINARY_DIR}/mpi_svd_labConfig.cmake"
        "${PROJECT_BINARY_DIR}/mpi_svd_labConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mpi_svd_lab"
)
