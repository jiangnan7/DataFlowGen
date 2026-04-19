llvm_canonicalize_cmake_booleans(
  HETEACC_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(HETEACC_TEST_DEPENDS
  FileCheck
  count
  not
  heteacc-opt
)

add_lit_testsuite(check-heteacc "Running the Heteacc regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${HETEACC_TEST_DEPENDS}
)
set_target_properties(check-heteacc PROPERTIES FOLDER "Tests")

add_lit_testsuites(HETEACC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HETEACC_TEST_DEPENDS})
