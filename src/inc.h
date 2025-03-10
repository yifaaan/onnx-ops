#pragma once

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

std::vector<Box> multiClassNMS(std::vector<Box>& boxes, float iou_threshold);

float computeIoU(const Box& box1, const Box& box2);
