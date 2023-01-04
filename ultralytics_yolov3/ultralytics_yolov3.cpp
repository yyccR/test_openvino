
#include <iostream>
#include "memory"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "openvino/runtime/runtime.hpp"

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };


void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}


cv::Mat resize_padding(cv::Mat &origin_image, float &ratio, int target_h, int target_w, int constant_value) {
    cv::Mat out_image(target_h, target_w, CV_8UC3);
    int width = origin_image.cols, height = origin_image.rows;
    ratio = std::min((float)target_w /(float)width, (float)target_h / (float)height);

    if(width == target_w && height == target_h)
    {
        origin_image.copyTo(out_image);
        return out_image;
    }

    memset(out_image.data, constant_value, target_h * target_w * 3);
    int new_width = (int)(ratio * (float )width), new_height = (int)(ratio * (float )height);

    cv::Mat resize_image;
    cv::resize(origin_image, resize_image, cv::Size(new_width, new_height));
    // 深拷贝, resize_image替换rect_image的同时,也会替换out_image中的对应部分
    cv::Mat rect_image = out_image(cv::Rect(0, 0, new_width, new_height));
    resize_image.copyTo(rect_image);
    return out_image;
}

int main() {

    std::string model_file("/Users/yang/CLionProjects/test_openvino/ultralytics_yolov3/fp16/yolov3-tiny-sim.xml");
    std::string image_file("/Users/yang/CLionProjects/test_openvino/images/traffic_road.jpg");

    cv::Mat image = cv::imread(image_file);
//    cv::Mat input_image;
//    cv::resize(image, input_image, cv::Size(640,640));
//    input_image.convertTo(input_image, CV_32F, 1.0/255.0);
//    float w_scale = (float) image.cols / 640.f;
//    float h_scale = (float) image.rows / 640.f;

    float ratio = 0.0;
    cv::Mat input_image = resize_padding(image, ratio, 640, 640, 114);
    input_image.convertTo(input_image, CV_32F, 1.0/255.0);

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_file);
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor by index
    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    std::cout << "input type: " << input_tensor.get_element_type() << std::endl;
    std::cout << "input shape: " << input_tensor.get_shape().to_string() << std::endl;
    // Element types, names and layouts are aligned with framework
    float* data = input_tensor.data<float>();

    cv::Mat image_channels[3];
    cv::split(input_image, image_channels);
    for (int j = 0; j < 3; j++) {
        memcpy(data + 640*640 * j, image_channels[j].data,640*640 * sizeof(float));
    }

    infer_request.infer();
    // model has only one output
    ov::Tensor output_tensor = infer_request.get_output_tensor(0);
    std::cout << "output type: " << output_tensor.get_element_type() << std::endl;
    std::cout << "output shape: " << output_tensor.get_shape().at(1) << std::endl;
    float* out_data = output_tensor.data<float>();

    std::vector<BoxInfo> boxes;
    int num_classes = 80;
    float score_threshold = 0.1;
    for(int i = 0; i < output_tensor.get_shape().at(1); i++){
        float obj_conf = *(out_data + (i * (num_classes + 5)) + 4);

        float cls_conf = *(out_data + (i * (num_classes + 5))+5);
        int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
//            float tmp_conf = offset_obj_cls_ptr[j + 5];
            float tmp_conf = *(out_data + (i * (num_classes + 5))+j+5);
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax

        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold) continue; //
        std::cout << "obj_conf " << obj_conf << " cls_conf " << cls_conf << " conf: " << conf << std::endl;

        float cx = *(out_data + (i * (num_classes + 5))+0);
//        float cy = offset_obj_cls_ptr[1];
        float cy = *(out_data + (i * (num_classes + 5))+1);
//        float w = output + (i * (num_classes + 5))[2];
        float w = *(out_data + (i * (num_classes + 5))+2);
//        float h = offset_obj_cls_ptr[3];
        float h = *(out_data + (i * (num_classes + 5))+3);
//        float x1 = (cx - w / 2.f) * w_scale;
//        float y1 = (cy - h / 2.f) * h_scale;
//        float x2 = (cx + w / 2.f) * w_scale;
//        float y2 = (cy + h / 2.f) * h_scale;
        float x1 = (cx - w / 2.f) / ratio;
        float y1 = (cy - h / 2.f) / ratio;
        float x2 = (cx + w / 2.f) / ratio;
        float y2 = (cy + h / 2.f) / ratio;

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) image.cols - 1.f);
        box.y2 = std::min(y2, (float) image.rows - 1.f);
        box.score = conf;
        box.label = label;
        boxes.push_back(box);

    }

    nms(boxes, 0.2);
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);

    return 0;
}