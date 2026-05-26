include_guard(GLOBAL)

function(mpilab_enable_sanitizers target sanitizers)
    if(MSVC)
        if(sanitizers STREQUAL "address")
            target_compile_options(${target} PRIVATE /fsanitize=address)
        else()
            message(WARNING "MSVC sanitizer support is limited; requested '${sanitizers}'.")
        endif()
        return()
    endif()

    if("thread" IN_LIST sanitizers AND ("address" IN_LIST sanitizers OR "leak" IN_LIST sanitizers))
        message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer or LeakSanitizer.")
    endif()

    list(JOIN sanitizers "," sanitizer_list)
    target_compile_options(${target} PRIVATE
        -fsanitize=${sanitizer_list}
        -fno-omit-frame-pointer
    )
    target_link_options(${target} PRIVATE
        -fsanitize=${sanitizer_list}
        -fno-omit-frame-pointer
    )
endfunction()

function(mpilab_enable_coverage target)
    if(MSVC)
        message(WARNING "Coverage instrumentation is not configured for MSVC.")
        return()
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Coverage is intended for Debug builds; current CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'.")
    endif()

    target_compile_options(${target} PRIVATE
        --coverage
        -O0
        -g
    )
    target_link_options(${target} PRIVATE --coverage)
endfunction()
