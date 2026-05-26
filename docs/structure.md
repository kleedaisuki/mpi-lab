mpi-svd-lab/
├── CMakeLists.txt
├── pyproject.toml
├── .gitignore
│
├── include/
│   └── mpilab/
│       ├── infrastructure/
│       │   ├── file_stream.hpp
│       │   ├── mpi_runtime.hpp
│       │   └── logger.hpp
│       │
│       ├── domain/
│       │   ├── matrix.hpp
│       │   ├── kernel.hpp
│       │   ├── MatrixLike.hpp
│       │   └── KernelLike.hpp
│       │
│       ├── application/
│       │   ├── pipeline.hpp
│       │   └── mpi_scheduler.hpp
│       │
│       └── presentation/
│
├── src/
│   └── mpilab/
│       ├── CMakeLists.txt
│       │
│       ├── infrastructure/
│       │   ├── file_stream.cpp
│       │   ├── mpi_runtime.cpp
│       │   └── logger.cpp
│       │
│       ├── domain/
│       │   ├── matrix/
│       │   └── kernel/
│       │
│       ├── application/
│       │   ├── pipeline.cpp
│       │   └── mpi_scheduler.cpp
│       │
│       └── presentation/
