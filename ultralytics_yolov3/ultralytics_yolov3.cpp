
#include <iostream>
#include "openvino/runtime/runtime.hpp"


int main() {

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("/Users/yang/CLionProjects/test_openvino/ultralytics_yolov3/yolov3-tiny-sim.xml");
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor by index
    ov::Tensor input_tensor1 = infer_request.get_input_tensor(0);
    std::cout << input_tensor1.get_shape().to_string() << std::endl;
    // Element types, names and layouts are aligned with framework
    auto data1 = input_tensor1.data<float>();
    // Fill first data ...

    // Get input tensor by tensor name
//    ov::Tensor input_tensor2 = infer_request.get_tensor("data2_t");
    // Element types, names and layouts are aligned with framework
//    auto data2 = input_tensor1.data<int64_t>();
    // Fill first data ...
    return 0;
}