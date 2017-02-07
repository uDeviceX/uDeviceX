add_executable(iotags iotags.cpp)
target_link_libraries (iotags LINK_PUBLIC geom-wrapper)

# use `auto' in c++
target_compile_features(iotags PRIVATE cxx_auto_type)
