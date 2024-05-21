#pragma once
#include "pch.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <map>
#include <vector>
#include <iostream>

static const int threshold_length = 10;
static const float threshold_dist = 1.414213562f;
static const double canny_th1 = 50.0;
static const double canny_th2 = 50.0;
static const int canny_aperture_size = 3;

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

class ximgproc{

private:

    static int guo_hall_iteration(const unsigned char* binary_image, unsigned char* mask, const unsigned int width, const unsigned int height, const int iteration) {
        /* one iteration of the algorithm by guo and hall. see their paper for an explanation.
           We only consider nonzero elemets of the image. We never reinitialize the mask, once a pixel is
           black, it will never become white again anyway. */
        unsigned int changed = 0;

        #pragma omp parallel for
        for (int n = 1; n < (int)height - 1; n++) {
            unsigned int j = (unsigned int)n;
            const unsigned char* line = binary_image + j * width;
            unsigned int start = 0;
            const unsigned int len = width - 1;

            while (start + 1 < len) {
                start = nonzero_clever(line, start + 1, len);
                if (start == len) break;

                const unsigned int i = start;

                const bool p2 = binary_image[i - 1 + width * j];
                const bool p6 = binary_image[i + 1 + width * j];

                const bool p9 = binary_image[i - 1 + width * (j - 1)];
                const bool p8 = binary_image[i + width * (j - 1)];
                const bool p7 = binary_image[i + 1 + width * (j - 1)];

                const bool p3 = binary_image[i - 1 + width * (j + 1)];
                const bool p4 = binary_image[i + width * (j + 1)];
                const bool p5 = binary_image[i + 1 + width * (j + 1)];
                const unsigned int C = ((!p2 && (p3 || p4)) +
                    (!p4 && (p5 || p6)) +
                    (!p6 && (p7 || p8)) +
                    (!p8 && (p9 || p2)));

                if (C == 1) {
                    const unsigned int N1 = (p9 || p2) + (p3 || p4) + (p5 || p6) + (p7 || p8);
                    const unsigned int N2 = (p2 || p3) + (p4 || p5) + (p6 || p7) + (p8 || p9);
                    const unsigned int N = N1 < N2 ? N1 : N2;
                    unsigned int m;

                    if (iteration == 0)
                    {
                        m = (p8 && (p6 || p7 || !p9));
                    }
                    else
                    {
                        m = (p4 && (p2 || p3 || !p5));
                    }

                    if (2 <= N && N <= 3 && m == 0) {
                        mask[i + width * j] = 0;
                        changed += 1;
                    }
                }
            }

        }
        return changed;
    }

    static int nonzero_clever(const unsigned char* arr, unsigned int start, unsigned int len) {
        /* find the first nonzero element from arr[start] to arr[start+len-1] (inclusive)
           look at a long long at a time to be faster on 64 bit cpus */
        const unsigned int step = sizeof(unsigned long long) / sizeof(unsigned char);
        unsigned int i = start;
        //unsigned types should throw exceptions on under/overflow...
        while (len > step && i < len - step) {
            if (*((unsigned long long*)(arr + i)) == 0) {
                i += step;
            }
            else {
                int j = 0;
                while (arr[i + j] == 0) j++;
                return i + j;
            }
        }
        while (i < len) {
            if (arr[i] != 0) { return i; }
            i++;
        }
        return len;
    }

    static void andImage_fast(unsigned char* image, const unsigned char* mask, const unsigned int size) {
        /* calculate image &=mask.
           to be faster on 64 bit cpus, we do this one long long at a time */
        const unsigned int step = sizeof(unsigned long long) / sizeof(unsigned char);
        unsigned long long* image_l = (unsigned long long*)image;
        const unsigned long long* mask_l = (unsigned long long*) mask;
        unsigned int i = 0;
        for (; size / step > 2 && i < size / step - 2; i += 2) {
            image_l[i] = image_l[i] & mask_l[i];
            image_l[i + 1] = image_l[i + 1] & mask_l[i + 1];
        }
        for (i = i * step; i < size; ++i) {
            image[i] = image[i] & mask[i];
        }
    }

    static void orImage_fast(unsigned char* image, const unsigned char* mask, const unsigned int size) {
        /* calculate image &=mask.
           to be faster on 64 bit cpus, we do this one long long at a time */
        const unsigned int step = sizeof(unsigned long long) / sizeof(unsigned char);
        unsigned long long* image_l = (unsigned long long*)image;
        const unsigned long long* mask_l = (unsigned long long*) mask;
        unsigned int i = 0;
        for (; size / step > 2 && i < size / step - 2; i += 2) {
            image_l[i] = image_l[i] | mask_l[i];
            image_l[i + 1] = image_l[i + 1] | mask_l[i + 1];
        }
        for (i = i * step; i < size; ++i) {
            image[i] = image[i] | mask[i];
        }
    }

    struct SEGMENT
    {
        double x1, y1, x2, y2, angle;
    };

    static bool getPointChain(const cv::Mat& img, cv::Point pt, cv::Point& chained_pt, double& direction, int step)
    {
        int ri, ci;
        int indices[8][2] = { {1,1}, {1,0}, {1,-1}, {0,-1},
            {-1,-1},{-1,0}, {-1,1}, {0,1} };

        double min_dir_diff = 7.0;
        cv::Point consistent_pt;
        int consistent_direction = 0;
        for (int i = 0; i < 8; i++)
        {
            ci = pt.x + indices[i][1];
            ri = pt.y + indices[i][0];

            if (ri < 0 || ri == img.rows || ci < 0 || ci == img.cols)
                continue;

            if (img.at<unsigned char>(ri, ci) == 0)
                continue;

            if (step == 0)
            {
                chained_pt.x = ci;
                chained_pt.y = ri;
                // direction = (float)i;
                direction = i > 4 ? (double)(i - 8) : (double)i;
                return true;
            }
            else
            {
                double curr_dir = i > 4 ? (double)(i - 8) : (double)i;
                double dir_diff = abs(curr_dir - direction);
                dir_diff = dir_diff > 4.0 ? 8.0 - dir_diff : dir_diff;
                if (dir_diff <= min_dir_diff)
                {
                    min_dir_diff = dir_diff;
                    consistent_pt.x = ci;
                    consistent_pt.y = ri;
                    consistent_direction = i > 4 ? i - 8 : i;
                }
            }
        }
        if (min_dir_diff < 2.0)
        {
            chained_pt.x = consistent_pt.x;
            chained_pt.y = consistent_pt.y;
            direction = (direction * (double)step + (double)consistent_direction) / (double)(step + 1);
            return true;
        }
        return false;
    }

    static void extractSegments(const std::vector<cv::Point2i>& points, std::vector<SEGMENT>& segments, const int imagewidth, const int imageheight)
    {
        bool is_line;

        int i, j;
        SEGMENT seg;
        cv::Point2i ps, pe, pt;

        std::vector<cv::Point2i> l_points;

        int total = (int)points.size();

        for (i = 0; i + threshold_length < total; i++)
        {
            ps = points[i];
            pe = points[i + threshold_length];

            double a[] = { (double)ps.x, (double)ps.y, 1 };
            double b[] = { (double)pe.x, (double)pe.y, 1 };
            double c[3], d[3];

            cv::Mat p1 = cv::Mat(3, 1, CV_64FC1, a).clone();
            cv::Mat p2 = cv::Mat(3, 1, CV_64FC1, b).clone();
            cv::Mat p = cv::Mat(3, 1, CV_64FC1, c).clone();
            cv::Mat l = cv::Mat(3, 1, CV_64FC1, d).clone();
            l = p1.cross(p2);

            is_line = true;

            l_points.clear();
            l_points.push_back(ps);

            for (j = 1; j < threshold_length; j++)
            {
                pt.x = points[i + j].x;
                pt.y = points[i + j].y;

                p.at<double>(0, 0) = (double)pt.x;
                p.at<double>(1, 0) = (double)pt.y;
                p.at<double>(2, 0) = 1.0;

                double dist = distPointLine(p, l);

                if (fabs(dist) > threshold_dist)
                {
                    is_line = false;
                    break;
                }
                l_points.push_back(pt);
            }

            // Line check fail, test next point
            if (is_line == false)
                continue;

            l_points.push_back(pe);

            cv::Vec4f line;
            fitLine(cv::Mat(l_points), line, cv::DIST_L2, 0, 0.01, 0.01);
            a[0] = line[2];
            a[1] = line[3];
            b[0] = line[2] + line[0];
            b[1] = line[3] + line[1];

            p1 = cv::Mat(3, 1, CV_64FC1, a).clone();
            p2 = cv::Mat(3, 1, CV_64FC1, b).clone();

            l = p1.cross(p2);

            incidentPoint(l, ps, imagewidth, imageheight);

            // Extending line
            for (j = threshold_length + 1; i + j < total; j++)
            {
                pt.x = points[i + j].x;
                pt.y = points[i + j].y;

                p.at<double>(0, 0) = (double)pt.x;
                p.at<double>(1, 0) = (double)pt.y;
                p.at<double>(2, 0) = 1.0;

                double dist = distPointLine(p, l);
                if (fabs(dist) > threshold_dist)
                {
                    fitLine(cv::Mat(l_points), line, cv::DIST_L2, 0, 0.01, 0.01);
                    a[0] = line[2];
                    a[1] = line[3];
                    b[0] = line[2] + line[0];
                    b[1] = line[3] + line[1];

                    p1 = cv::Mat(3, 1, CV_64FC1, a).clone();
                    p2 = cv::Mat(3, 1, CV_64FC1, b).clone();

                    l = p1.cross(p2);
                    dist = distPointLine(p, l);
                    if (fabs(dist) > threshold_dist) {
                        j--;
                        break;
                    }
                }
                pe = pt;
                l_points.push_back(pt);
            }
            fitLine(cv::Mat(l_points), line, cv::DIST_L2, 0, 0.01, 0.01);
            a[0] = line[2];
            a[1] = line[3];
            b[0] = line[2] + line[0];
            b[1] = line[3] + line[1];

            p1 = cv::Mat(3, 1, CV_64FC1, a).clone();
            p2 = cv::Mat(3, 1, CV_64FC1, b).clone();

            l = p1.cross(p2);

            cv::Point2f e1, e2;
            e1.x = (float)ps.x;
            e1.y = (float)ps.y;
            e2.x = (float)pe.x;
            e2.y = (float)pe.y;

            incidentPoint(l, e1, imagewidth, imageheight);
            incidentPoint(l, e2, imagewidth, imageheight);
            seg.x1 = e1.x;
            seg.y1 = e1.y;
            seg.x2 = e2.x;
            seg.y2 = e2.y;

            segments.push_back(seg);
            i = i + j;
        }
    }

    static void mergeLines(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
    {
        double xg = 0.0, yg = 0.0;
        double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
        double ax = 0, bx = 0, cx = 0, dx = 0;
        double ay = 0, by = 0, cy = 0, dy = 0;
        double li = 0.0, lj = 0.0;
        double thi = 0.0, thj = 0.0, thr = 0.0;
        double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

        ax = seg1.x1;
        ay = seg1.y1;

        bx = seg1.x2;
        by = seg1.y2;
        cx = seg2.x1;
        cy = seg2.y1;

        dx = seg2.x2;
        dy = seg2.y2;

        double dlix = (bx - ax);
        double dliy = (by - ay);
        double dljx = (dx - cx);
        double dljy = (dy - cy);

        li = sqrt((dlix * dlix) + (dliy * dliy));
        lj = sqrt((dljx * dljx) + (dljy * dljy));

        xg = (li * (ax + bx) + lj * (cx + dx))
            / (2.0 * (li + lj));
        yg = (li * (ay + by) + lj * (cy + dy))
            / (2.0 * (li + lj));

        if (dlix == 0.0) thi = CV_PI / 2.0;
        else thi = atan(dliy / dlix);

        if (dljx == 0.0) thj = CV_PI / 2.0;
        else thj = atan(dljy / dljx);

        if (fabs(thi - thj) <= CV_PI / 2.0)
        {
            thr = (li * thi + lj * thj) / (li + lj);
        }
        else
        {
            double tmp = thj - CV_PI * (thj / fabs(thj));
            thr = li * thi + lj * tmp;
            thr /= (li + lj);
        }

        axg = (ay - yg) * sin(thr) + (ax - xg) * cos(thr);
        bxg = (by - yg) * sin(thr) + (bx - xg) * cos(thr);
        cxg = (cy - yg) * sin(thr) + (cx - xg) * cos(thr);
        dxg = (dy - yg) * sin(thr) + (dx - xg) * cos(thr);

        delta1xg = min(axg, min(bxg, min(cxg, dxg)));
        delta2xg = max(axg, max(bxg, max(cxg, dxg)));

        delta1x = delta1xg * cos(thr) + xg;
        delta1y = delta1xg * sin(thr) + yg;
        delta2x = delta2xg * cos(thr) + xg;
        delta2y = delta2xg * sin(thr) + yg;

        seg_merged.x1 = delta1x;
        seg_merged.y1 = delta1y;
        seg_merged.x2 = delta2x;
        seg_merged.y2 = delta2y;
    }

    static double distPointLine(const cv::Mat& p, cv::Mat& l)
    {
        double x = l.at<double>(0, 0);
        double y = l.at<double>(1, 0);
        double w = sqrt(x * x + y * y);

        l.at<double>(0, 0) = x / w;
        l.at<double>(1, 0) = y / w;
        l.at<double>(2, 0) = l.at<double>(2, 0) / w;

        return l.dot(p);
    }

    static bool mergeSegments(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
    {
        double o[] = { 0.0, 0.0, 1.0 };
        double a[] = { 0.0, 0.0, 1.0 };
        double b[] = { 0.0, 0.0, 1.0 };
        double c[3];

        o[0] = (seg2.x1 + seg2.x2) / 2.0;
        o[1] = (seg2.y1 + seg2.y2) / 2.0;

        a[0] = seg1.x1;
        a[1] = seg1.y1;
        b[0] = seg1.x2;
        b[1] = seg1.y2;

        cv::Mat ori = cv::Mat(3, 1, CV_64FC1, o).clone();
        cv::Mat p1 = cv::Mat(3, 1, CV_64FC1, a).clone();
        cv::Mat p2 = cv::Mat(3, 1, CV_64FC1, b).clone();
        cv::Mat l1 = cv::Mat(3, 1, CV_64FC1, c).clone();

        l1 = p1.cross(p2);

        cv::Point2d seg1mid, seg2mid;
        seg1mid.x = (seg1.x1 + seg1.x2) / 2.0;
        seg1mid.y = (seg1.y1 + seg1.y2) / 2.0;
        seg2mid.x = (seg2.x1 + seg2.x2) / 2.0;
        seg2mid.y = (seg2.y1 + seg2.y2) / 2.0;

        double seg1len = sqrt((seg1.x1 - seg1.x2) * (seg1.x1 - seg1.x2) + (seg1.y1 - seg1.y2) * (seg1.y1 - seg1.y2));
        double seg2len = sqrt((seg2.x1 - seg2.x2) * (seg2.x1 - seg2.x2) + (seg2.y1 - seg2.y2) * (seg2.y1 - seg2.y2));
        double middist = sqrt((seg1mid.x - seg2mid.x) * (seg1mid.x - seg2mid.x) + (seg1mid.y - seg2mid.y) * (seg1mid.y - seg2mid.y));
        double angdiff = fabs(seg1.angle - seg2.angle);

        double dist = distPointLine(ori, l1);

        if (fabs(dist) <= threshold_dist * 2.0 && middist <= seg1len / 2.0 + seg2len / 2.0 + 20.0
            && angdiff <= CV_PI / 180.0 * 5.0)
        {
            mergeLines(seg1, seg2, seg_merged);
            return true;
        }
        else
        {
            return false;
        }
    }

    static void pointInboardTest(const cv::Size srcSize, cv::Point2i& pt)
    {
        pt.x = pt.x <= 5 ? 5 : pt.x >= srcSize.width - 5 ? srcSize.width - 5 : pt.x;
        pt.y = pt.y <= 5 ? 5 : pt.y >= srcSize.height - 5 ? srcSize.height - 5 : pt.y;
    }

    template<class T>
    static void incidentPoint(const cv::Mat& l, T& pt, const int imagewidth, const int imageheight)
    {
        double a[] = { (double)pt.x, (double)pt.y, 1.0 };
        double b[] = { l.at<double>(0,0), l.at<double>(1,0), 0.0 };
        double c[3];

        cv::Mat xk = cv::Mat(3, 1, CV_64FC1, a).clone();
        cv::Mat lh = cv::Mat(3, 1, CV_64FC1, b).clone();
        cv::Mat lk = cv::Mat(3, 1, CV_64FC1, c).clone();

        lk = xk.cross(lh);
        xk = lk.cross(l);

        xk.convertTo(xk, -1, 1.0 / xk.at<double>(2, 0));

        cv::Point2d pt_tmp;
        pt_tmp.x = xk.at<double>(0, 0) < 0.0 ? 0.0 : xk.at<double>(0, 0)
            >= (imagewidth - 1.0) ? (imagewidth - 1.0) : xk.at<double>(0, 0);
        pt_tmp.y = xk.at<double>(1, 0) < 0.0 ? 0.0 : xk.at<double>(1, 0)
            >= (imageheight - 1.0) ? (imageheight - 1.0) : xk.at<double>(1, 0);
        pt = T(pt_tmp);
    }

    static inline void getAngle(SEGMENT& seg)
    {
        seg.angle = (double)cv::fastAtan2((float)(seg.y2 - seg.y1), (float)(seg.x2 - seg.x1)) / 180.0 * CV_PI;
    }

    static void additionalOperationsOnSegment(const cv::Mat& src, SEGMENT& seg)
    {
        if (seg.x1 == 0.0 && seg.x2 == 0.0 && seg.y1 == 0.0 && seg.y2 == 0.0)
            return;

        getAngle(seg);
        double ang = seg.angle;

        cv::Point2d start = cv::Point2d(seg.x1, seg.y1);
        cv::Point2d end = cv::Point2d(seg.x2, seg.y2);

        double dx = 0.0, dy = 0.0;
        dx = end.x - start.x;
        dy = end.y - start.y;

        int num_points = 10;
        cv::Point2d* points = new cv::Point2d[num_points];

        points[0] = start;
        points[num_points - 1] = end;
        for (int i = 0; i < num_points; i++)
        {
            if (i == 0 || i == num_points - 1)
                continue;
            points[i].x = points[0].x + (dx / double(num_points - 1) * i);
            points[i].y = points[0].y + (dy / double(num_points - 1) * i);
        }

        cv::Point2i* points_right = new cv::Point2i[num_points];
        cv::Point2i* points_left = new cv::Point2i[num_points];
        double gap = 1.0;

        for (int i = 0; i < num_points; i++)
        {
            points_right[i].x = cvRound(points[i].x + gap * cos(90.0 * CV_PI / 180.0 + ang));
            points_right[i].y = cvRound(points[i].y + gap * sin(90.0 * CV_PI / 180.0 + ang));
            points_left[i].x = cvRound(points[i].x - gap * cos(90.0 * CV_PI / 180.0 + ang));
            points_left[i].y = cvRound(points[i].y - gap * sin(90.0 * CV_PI / 180.0 + ang));
            pointInboardTest(src.size(), points_right[i]);
            pointInboardTest(src.size(), points_left[i]);
        }

        int iR = 0, iL = 0;
        for (int i = 0; i < num_points; i++)
        {
            iR += src.at<unsigned char>(points_right[i].y, points_right[i].x);
            iL += src.at<unsigned char>(points_left[i].y, points_left[i].x);
        }

        if (iR > iL)
        {
            std::swap(seg.x1, seg.x2);
            std::swap(seg.y1, seg.y2);
            getAngle(seg);
        }

        delete[] points;
        delete[] points_right;
        delete[] points_left;

        return;
    }

    static void lineDetection(const cv::Mat& src, std::vector<SEGMENT>& segments_all, const bool do_merge)
    {

        int r, c;
        int imageheight = src.rows; 
        int imagewidth = src.cols;

        std::vector<cv::Point2i> points;
        std::vector<SEGMENT> segments, segments_tmp;
        cv::Mat canny;
        if (canny_aperture_size == 0)
        {
            canny = src;
        }
        else
        {
            cv::Canny(src, canny, canny_th1, canny_th2, canny_aperture_size);
        }
        canny.colRange(0, 6).rowRange(0, 6).setTo(cv::Scalar::all(0));
        canny.colRange(src.cols - 5, src.cols).rowRange(src.rows - 5, src.rows).setTo(cv::Scalar::all(0));

        SEGMENT seg, seg1, seg2;

        for (r = 0; r < imageheight; r++)
        {
            for (c = 0; c < imagewidth; c++)
            {
                // Find seeds - skip for non-seeds
                if (canny.at<unsigned char>(r, c) == 0)
                    continue;

                // Found seeds
                cv::Point2i pt = cv::Point2i(c, r);

                points.push_back(pt);
                canny.at<unsigned char>(pt.y, pt.x) = 0;

                double direction = 0.0;
                int step = 0;
                while (getPointChain(canny, pt, pt, direction, step))
                {
                    points.push_back(pt);
                    step++;
                    canny.at<unsigned char>(pt.y, pt.x) = 0;
                }

                if (points.size() < (unsigned int)threshold_length + 1)
                {
                    points.clear();
                    continue;
                }

                extractSegments(points, segments, imagewidth, imageheight);

                if (segments.size() == 0)
                {
                    points.clear();
                    continue;
                }
                for (int i = 0; i < (int)segments.size(); i++)
                {
                    seg = segments[i];
                    double length = sqrt((seg.x1 - seg.x2) * (seg.x1 - seg.x2) +
                        (seg.y1 - seg.y2) * (seg.y1 - seg.y2));
                    if (length < threshold_length)
                        continue;
                    if ((seg.x1 <= 5.0 && seg.x2 <= 5.0) ||
                        (seg.y1 <= 5.0 && seg.y2 <= 5.0) ||
                        (seg.x1 >= imagewidth - 5.0 && seg.x2 >= imagewidth - 5.0) ||
                        (seg.y1 >= imageheight - 5.0 && seg.y2 >= imageheight - 5.0))
                        continue;
                    additionalOperationsOnSegment(src, seg);
                    if (!do_merge)
                        segments_all.push_back(seg);
                    segments_tmp.push_back(seg);
                }
                points.clear();
                segments.clear();
            }
        }
        if (!do_merge)
            return;

        bool is_merged = false;
        int ith = (int)segments_tmp.size() - 1;
        int jth = ith - 1;
        while (ith > 1 || jth > 0)
        {
            seg1 = segments_tmp[ith];
            seg2 = segments_tmp[jth];
            SEGMENT seg_merged;
            is_merged = mergeSegments(seg1, seg2, seg_merged);
            if (is_merged == true)
            {
                seg2 = seg_merged;
                additionalOperationsOnSegment(src, seg2);
                std::vector<SEGMENT>::iterator it = segments_tmp.begin() + ith;
                *it = seg2;
                segments_tmp.erase(segments_tmp.begin() + jth);
                ith--;
                jth = ith - 1;
            }
            else
            {
                jth--;
            }
            if (jth < 0) {
                ith--;
                jth = ith - 1;
            }
        }
        segments_all = segments_tmp;
    }

public:

    static void guo_hall_thinning(cv::Mat& image) {

        const cv::Size& s = image.size();
        const int width = s.width;
        const int height = s.height;

        unsigned char* binary_image = image.data;

        int changed;
        unsigned char* mask = (unsigned char*)malloc(width * height * sizeof(unsigned char));

        if (mask == NULL) {
            std::cout << "Error: Failed to allocate memory - Insufficient Memory";
            exit(1);
        }

        memset(mask, UCHAR_MAX, width * height);
        do {
            changed = guo_hall_iteration(binary_image, mask, width, height, 0);
            andImage_fast(binary_image, mask, width * height);

            changed += guo_hall_iteration(binary_image, mask, width, height, 1);
            andImage_fast(binary_image, mask, width * height);
        } while (changed != 0);
        free(mask);

    }

    static std::vector<std::array<double, 4>> fastLineDetector(cv::Mat& image, const bool do_merge)
    {

        std::vector<std::array<double, 4>> lines;
        std::vector<SEGMENT> segments;

        lineDetection(image, segments, do_merge);

        for (size_t i = 0; i < segments.size(); ++i)
        {
            const SEGMENT seg = segments[i];
            //cv::Vec4f line(seg.x1, seg.y1, seg.x2, seg.y2);
            lines.push_back({ seg.x1, seg.y1, seg.x2, seg.y2 });
        }
       
        return lines;
    }

    static void drawSegments(cv::Mat& image, std::vector<std::array<double, 4>>& lines, bool draw_arrow, cv::Scalar linecolor, int linethickness)
    {

        /*int cn = image.channels();
        CV_Assert(!image.empty() && (cn == 1 || cn == 3 || cn == 4));

        if (cn == 1)
        {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }
        else
        {
            cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
            cv::cvtColor(image, image, cn == 3 ? cv::COLOR_GRAY2BGR : cv::COLOR_GRAY2BGRA);
        }*/

        double gap = 10.0;
        double arrow_angle = 30.0;

        // Draw segments
        for (int i = 0; i < lines.size(); ++i)
        {
            const std::array<double, 4>& v = lines[i];
            cv::Point2d b(v[0], v[1]);
            cv::Point2d e(v[2], v[3]);
            cv::line(image, b, e, linecolor, linethickness);
            if (draw_arrow)
            {
                SEGMENT seg;
                seg.x1 = b.x;
                seg.y1 = b.y;
                seg.x2 = e.x;
                seg.y2 = e.y;
                getAngle(seg);
                double ang = (double)seg.angle;
                cv::Point2i p1;
                p1.x = cvRound(seg.x2 - gap * cos(arrow_angle * CV_PI / 180.0 + ang));
                p1.y = cvRound(seg.y2 - gap * sin(arrow_angle * CV_PI / 180.0 + ang));
                pointInboardTest(image.size(), p1);
                cv::line(image, cv::Point(cvRound(seg.x2), cvRound(seg.y2)), p1, linecolor, linethickness);
            }
        }
    }
};

