# Test Openvino in Macos

## Macos编译`libopenvino.dylib`动态库
```
git clone -b 2022.3.0 https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init


mkdir build_macos && cd build_macos

# 通过参数 -DTHREADING=SEQ 可以取消tbb依赖库
cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release --parallel $(sysctl -n hw.ncpu)
```

## 检测效果

[!检测效果](/images/det_image.jpg)