#include <vector>
#include <iostream>


// �����������IOU
float calculateIOU(const std::vector<float>& box1,
    const std::vector<float>& box2,
    bool center_point_box) {
    float x1, y1, x2, y2;  // box1������
    float x1_, y1_, x2_, y2_;  // box2������

    // ���ݸ�ʽ��������
    if (center_point_box) {
        // [x_center, y_center, width, height] ת��Ϊ [x1, y1, x2, y2]
        float w1 = box1[2], h1 = box1[3];
        if (w1 < 0 || h1 < 0) return 0.0f;  // ���Ȼ�߶�Ϊ������Ч��
        x1 = box1[0] - w1 / 2;
        y1 = box1[1] - h1 / 2;
        x2 = box1[0] + w1 / 2;
        y2 = box1[1] + h1 / 2;

        float w2 = box2[2], h2 = box2[3];
        if (w2 < 0 || h2 < 0) return 0.0f;  // ���Ȼ�߶�Ϊ������Ч��
        x1_ = box2[0] - w2 / 2;
        y1_ = box2[1] - h2 / 2;
        x2_ = box2[0] + w2 / 2;
        y2_ = box2[1] + h2 / 2;
    }
    else {
        // [y1, x1, y2, x2] ��ʽ
        x1 = box1[1]; y1 = box1[0];
        x2 = box1[3]; y2 = box1[2];
        x1_ = box2[1]; y1_ = box2[0];
        x2_ = box2[3]; y2_ = box2[2];
    }

    // ���������Ч��
    if (x1 > x2 || y1 > y2 || x1_ > x2_ || y1_ > y2_) {
        return 0.0f;  // ��Ч�����Ϊ������
    }

    // ���㽻������
    float xi1 = std::max(x1, x1_);  // �������Ͻ�x
    float yi1 = std::max(y1, y1_);  // �������Ͻ�y
    float xi2 = std::min(x2, x2_);  // �������½�x
    float yi2 = std::min(y2, y2_);  // �������½�y

    // ���㽻������
    float inter_width = std::max(0.0f, xi2 - xi1);
    float inter_height = std::max(0.0f, yi2 - yi1);
    float inter_area = inter_width * inter_height;

    // ����ÿ��������
    float box1_area = (x2 - x1) * (y2 - y1);
    float box2_area = (x2_ - x1_) * (y2_ - y1_);

    // �������Ƿ���Ч
    if (box1_area <= 0.0f || box2_area <= 0.0f) {
        return 0.0f;  // �������Ч
    }

    // ���㲢�����
    float union_area = box1_area + box2_area - inter_area;

    // ��鲢�����
    if (union_area <= 0.0f) {
        return 0.0f;  // ���������Ч
    }

    // ����IOU
    float iou = inter_area / union_area;

    // ȷ��IOU��[0,1]��Χ��
    if (iou < 0.0f || iou > 1.0f) {
        std::cerr << "Invalid IOU: " << iou << ", inter_area: " << inter_area
            << ", union_area: " << union_area << std::endl;
        return 0.0f;
    }

    return iou;
}






std::vector<std::vector<int64_t>> nonMaxSuppression(
    const std::vector<std::vector<std::vector<float>>>& boxes,
    const std::vector<std::vector<std::vector<float>>>& scores,
    int64_t max_output_boxes_per_class = 0,
    float iou_threshold = 0.0f,
    float score_threshold = 0.0f,
    int center_point_box = 0) {

    if (boxes.empty() || scores.empty()) {
        std::cerr << "Error: Empty input tensors" << std::endl;
        return {};
    }
    size_t num_batches = boxes.size();
    if (scores.size() != num_batches) {
        std::cerr << "Error: Mismatch in num_batches" << std::endl;
        return {};
    }
    if (boxes[0].empty() || scores[0].empty()) {
        std::cerr << "Error: Empty spatial dimension or classes" << std::endl;
        return {};
    }
    size_t spatial_dimension = boxes[0].size();
    size_t num_classes = scores[0].size();
    for (const auto& batch : boxes) {
        if (batch.size() != spatial_dimension || batch[0].size() != 4) {
            std::cerr << "Error: Invalid boxes dimension" << std::endl;
            return {};
        }
    }
    for (const auto& batch : scores) {
        if (batch.size() != num_classes || batch[0].size() != spatial_dimension) {
            std::cerr << "Error: Invalid scores dimension" << std::endl;
            return {};
        }
    }

    std::vector<std::vector<int64_t>> selected_indices;

    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        for (size_t class_idx = 0; class_idx < num_classes; class_idx++) {
            std::vector<std::pair<float, size_t>> score_pairs;

            for (size_t box_idx = 0; box_idx < spatial_dimension; box_idx++) {
                float score = scores[batch_idx][class_idx][box_idx];
                if (score >= score_threshold) {
                    score_pairs.emplace_back(score, box_idx);
                }
            }

            std::sort(score_pairs.begin(), score_pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            std::vector<bool> suppressed(spatial_dimension, false);
            size_t selected_count = 0;

            for (size_t i = 0; i < score_pairs.size() &&
                (max_output_boxes_per_class == 0 || selected_count < max_output_boxes_per_class); i++) {
                if (suppressed[score_pairs[i].second]) continue;

                size_t curr_idx = score_pairs[i].second;
                selected_indices.push_back({ (int64_t)batch_idx, (int64_t)class_idx, (int64_t)curr_idx });
                selected_count++;

                for (size_t j = i + 1; j < score_pairs.size(); j++) {
                    if (suppressed[score_pairs[j].second]) continue;

                    float iou = calculateIOU(
                        boxes[batch_idx][curr_idx],
                        boxes[batch_idx][score_pairs[j].second],
                        center_point_box != 0
                    );

                    // ��ӡ������Ϣ
                    // std::cout << "Comparing box " << curr_idx << " (score: " << score_pairs[i].first
                    //     << ") with box " << score_pairs[j].second << " (score: " << score_pairs[j].first
                    //     << "), IOU: " << iou << std::endl;

                    if (iou > iou_threshold) {
                        suppressed[score_pairs[j].second] = true;
                        // std::cout << "Comparing box " << curr_idx << " (score: " << score_pairs[i].first
                        //     << ") with box " << score_pairs[j].second << " (score: " << score_pairs[j].first
                        //     << "), IOU: " << iou << std::endl;
                        // std::cout << "Suppressing box " << score_pairs[j].second << std::endl;
                    }
                }
            }
        }
    }

    return selected_indices;
}


