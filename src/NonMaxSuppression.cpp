#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

struct Box
{
    int class_id;
    float x1, y1, x2, y2;
    float score;

    Box(int cls, float x1, float y1, float x2, float y2, float s)
        : class_id(cls), x1(x1), y1(y1), x2(x2), y2(y2), score(s)
    {
    }
};

// IoU���㺯��
float computeIoU(const Box& box1, const Box& box2)
{
    // ���㽻������
    float inter_x1 = std::max(box1.x1, box2.x1);
    float inter_y1 = std::max(box1.y1, box2.y1);
    float inter_x2 = std::min(box1.x2, box2.x2);
    float inter_y2 = std::min(box1.y2, box2.y2);

    float inter_width = std::max(0.0f, inter_x2 - inter_x1);
    float inter_height = std::max(0.0f, inter_y2 - inter_y1);
    float intersection = inter_width * inter_height;

    // ���㲢������
    float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    float union_area = box1_area + box2_area - intersection;

    return intersection / union_area; // IoU
}

std::vector<Box> multiClassNMS(std::vector<Box>& boxes, float iou_threshold)
{

    std::unordered_map<int, std::vector<Box>> class_groups;
    for (const auto& box : boxes)
    { // ����ͬ��������ֿ�
        class_groups[box.class_id].push_back(box);
    }

    std::vector<Box> result;

    for (auto& entry : class_groups)
    {
        int cls = entry.first;
        std::vector<Box>& group = entry.second;

        // �����߼�
        std::sort(group.begin(), group.end(),
                  [](const Box& a, const Box& b) { return a.score > b.score; });

        std::vector<Box> temp;
        while (!group.empty())
        {
            Box selected = group.front();
            temp.push_back(selected);
            group.erase(group.begin());

            auto it = group.begin();
            while (it != group.end())
            {
                if (computeIoU(selected, *it) > iou_threshold)
                {
                    it = group.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        result.insert(result.end(), temp.begin(), temp.end());
    }

    // ȫ������
    std::sort(result.begin(), result.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });

    return result;
}

// ��ӡ��Ľ��
void printBoxes(const std::vector<Box>& boxes)
{
    for (const auto& box : boxes)
    {
        std::cout << "Box: [" << box.class_id << "," << box.x1 << ", " << box.y1 << ", " << box.x2
                  << ", " << box.y2 << "], Score: " << box.score << std::endl;
    }
}

// int main() {

//     //��������1
//     std::cout << "CASE 1:" << std::endl;
//     std::vector<Box> boxes1 = {
//         Box(0, 50,50,200,200,0.9),
//         Box(0, 60,60,210,210,0.85),
//         Box(1, 250,250,400,400,0.8),
//         Box(0, 55,55,195,195,0.7),
//         Box(1, 255,255,405,405,0.75)
//     };

//     auto result1 = multiClassNMS(boxes1, 0.5);
//     // ��ӡ���

//     printBoxes(result1);
//     // Ԥ�������
//     // Class 0: [50,50,200,200] 0.9
//     // Class 1: [250,250,400,400] 0.8

//     //��������2
//     std::cout << "CASE 2:" << std::endl;
//     std::vector<Box> boxes2 = {
//         // ���0
//         Box(0, 100, 120, 150, 180, 0.95),
//         Box(0, 110, 130, 160, 185, 0.88),

//         // ���1
//         Box(1, 200, 300, 400, 450, 0.92),
//         Box(1, 210, 310, 390, 440, 0.85),

//         // ���2
//         Box(2, 50, 80, 120, 200, 0.90),
//         Box(2, 60, 85, 115, 195, 0.82)
//     };
//     auto result2 = multiClassNMS(boxes2, 0.5);
//     printBoxes(result2);
//     // Ԥ�������
//     /*  Class 0: [100, 120, 150, 180] 0.95
//         Class 1 : [200, 300, 400, 450] 0.92
//         Class 2 : [50, 80, 120, 200] 0.90*/
//     //��������3
//     std::cout << "CASE 3:" << std::endl;
//     std::vector<Box> boxes3 = {
//         // ���3
//         Box(3, 10,10,30,30,0.91),
//         Box(3, 15,15,35,35,0.89),
//         Box(3, 100,100,120,120,0.88),

//         // ���4
//         Box(4, 200,200,220,220,0.87),
//         Box(4, 202,202,222,222,0.86)
//     };

//     auto result3 = multiClassNMS(boxes3, 0.4);
//     printBoxes(result3);
//     //Class 3: [10, 10, 30, 30] 0.91
//     //Class 3: [15, 15, 35, 35] 0.89
//     //Class 3 : [100, 100, 120, 120] 0.88
//     //Class 4 : [200, 200, 220, 220] 0.87

//     //��������4
//     std::cout << "CASE 4:" << std::endl;
//     std::vector<Box> boxes4 = {
//         // ���5
//         Box(5, 50,50,250,250,0.93),

//         // ���6
//         Box(6, 60,60,240,240,0.91),

//         // ͬ���ص�
//         Box(6, 200,200,300,300,0.88),
//         Box(6, 210,210,310,310,0.85)
//     };

//     auto result4 = multiClassNMS(boxes4, 0.5);
//     printBoxes(result4);
//     /*Class 5: [50, 50, 250, 250] 0.93
//     Class 6 : [60, 60, 240, 240] 0.91
//     Class 6 : [200, 200, 300, 300] 0.88*/

// }