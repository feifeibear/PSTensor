add_library(ps_core
            OBJECT
            tensor.cpp
            cache_allocator.cpp
            base_allocator.cpp
            global_allocator.cpp
        )
target_link_libraries(ps_core PUBLIC
        dlpack
        ${RT_LIBRARY}
        ${CMAKE_DL_LIBS}
        cudart cuda cublas
        )