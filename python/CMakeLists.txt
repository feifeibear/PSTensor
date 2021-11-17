pybind11_add_module(ps_tensor_cxx pybind.cpp)

target_link_libraries(ps_tensor_cxx PRIVATE core)

SET(PY_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pypackage)
# file(GLOB_RECURSE PY_PROJ_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${PY_PACKAGE_DIR}/setup.py @ONLY)

add_custom_command(OUTPUT ${PY_PACKAGE_DIR}/.timestamp
        # COMMAND "${CMAKE_COMMAND}" -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/core ${PY_PACKAGE_DIR}/core
        COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:ps_tensor_cxx>"
        "${PY_PACKAGE_DIR}/ps_tensor/$<TARGET_FILE_NAME:ps_tensor_cxx>"
        COMMAND env CC=${CMAKE_C_COMPILER} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E touch ${PY_PACKAGE_DIR}/.timestamp
        WORKING_DIRECTORY ${PY_PACKAGE_DIR}
        # DEPENDS ps_tensor_cxx ${PY_PROJ_FILES}
        )
add_custom_target(ps_tensor ALL DEPENDS ${PY_PACKAGE_DIR}/.timestamp)

file(GLOB PY_TEST_FILES tests/*_test.py)

# foreach (py_test_file ${PY_TEST_FILES})
#     get_filename_component(py_test_trg_name ${py_test_file} NAME_WE)
#     message(STATUS "Add python test ${py_test_file}")
#     add_test(NAME
#             ${py_test_trg_name}
#             COMMAND
#             ${CMAKE_COMMAND} -E env
#             PYTHONPATH=${PY_PACKAGE_DIR}
#             ${PYTHON_EXECUTABLE} ${py_test_file}
#             WORKING_DIRECTORY
#             ${CMAKE_CURRENT_BINARY_DIR})
# endforeach ()
