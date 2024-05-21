#pragma once
#include "pch.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>
#include <map>


namespace JoinLines {
	constexpr double PI = 3.141592653589793;

    //shorthand for calculating the distance squared, it is faster to only calculate the distance squared than the distance and 
    //commonly the true distance is not needed for comparisons (For example is dist A > dist B is equivalent to (dist A)^2 > (dist B)^2)
    //x = difference in x coords
    //y = difference in y coords
    double inline distSquared(const double x, const double y) {
        return x * x + y * y;
    }

    //finds the pair of points between two lines (one point belonging to each line) with the longest distance apart
    //returns a new line between the longest point pair
    std::array<double, 4> longestLinesPointsPair(const double p1[2], const double p2[2], const double p3[2], const double p4[2]) {

        //distance squared between each possible pair
        const double A = distSquared(p1[0] - p3[0], p1[1] - p3[1]);
        const double B = distSquared(p1[0] - p4[0], p1[1] - p4[1]);
        const double C = distSquared(p2[0] - p3[0], p2[1] - p3[1]);
        const double D = distSquared(p2[0] - p4[0], p2[1] - p4[1]);

        const double maxDist = std::max(A, std::max(B, std::max(C, D)));

        //new line to return
        const std::array<double, 4> l = maxDist == A ? std::array<double, 4>{p1[0], p1[1], p3[0], p3[1]} :
                                        maxDist == B ? std::array<double, 4>{p1[0], p1[1], p4[0], p4[1]} :
                                        maxDist == C ? std::array<double, 4>{p2[0], p2[1], p3[0], p3[1]} :
                                        std::array<double, 4>{p2[0], p2[1], p4[0], p4[1]};

        //ensure the second point of the line has the larger x coord
        //l = l[0] > l[2] ? std::array<double, 4>{l[2], l[3], l[0], l[1]} : std::array<double, 4>{l[0], l[1], l[2], l[3]};

        return l;
    }

    double distToSegmentSquared(const std::array<double, 4>& l, const double p[2]) {

        const double l2 = distSquared(l[0] - l[2], l[1] - l[3]);

        if (l2 < 1e-7) return distSquared(l[0] - p[0], l[1] - p[1]);

        double t = ((p[0] - l[0]) * (l[2] - l[0]) + (p[1] - l[1]) * (l[3] - l[1])) / l2;

        t = std::max(0.0, std::min(1.0, t));

        return distSquared(p[0] - (l[0] + t * (l[2] - l[0])), p[1] - (l[1] + t * (l[3] - l[1])));
    }


    //Function for joining a set of lines from the output of an edge based line detection algorithm (openCV fast line detector).
    //Joins the lines based on having similar angles, the distance from the endpoint of one line to another line segment being 
    //less than a threshold, and/or, the distance and angle between two connecting endpoint (first to last or last to first) 
    //being again within thresholds.
    //For performance, rather than checking every line against every other line, the lines are first sorted into grid squares of 
    //size 256x256 and only compared to lines in or around the same grid square.
    //
    //lines = list of lines and their corresponding endpoints
    //x = number of horizontal grid squares (ceil(image width / 256))
    //y = number of vertical grid squares (ceil(image height / 256))
    //angleThreshold = max angle difference between two lines to be considered as potentially the same line
    //directMergeAngleThreshold = half the search arc angle for 'directly' merging the lines
    //directMergeThreshold = search arc radius for 'directly' merging the lines
    //distThreshold = max distance between an endpoint and oposite line segment to be considered the same line
    //minLineLength = after all lines are joined, any line smaller than this value is removed
    //maxLineLength = after all lines are joined, any line larger than this value is removed
    std::vector<std::array<double, 4>> joinLines(std::vector<std::array<double, 4>>& lines, const int x, const int y, const double angleThreshold, const double directMergeAngleThreshold, double directMergeThreshold, double distThreshold, double minLineLength, double maxLineLength) {
        //precalculate the angles of each line for performance
        std::vector<double> lineAngles;
        lineAngles.reserve(lines.size());

        //2d grid of vectors to store which lines are in each grid
        std::vector<std::vector<std::vector<int>>> lineGrid;

        for (int i = 0; i < y; i++) {
            lineGrid.emplace_back();
            for (int j = 0; j < x; j++) {
                lineGrid.at(i).emplace_back();
            }
        }

        //pre-calculations for performance reasons
        for (int i = 0; i < (int)lines.size(); i++) {
            auto line = lines.at(i);

            //ensure the second point of the line has the larger x coord
            //this makes sure the angle of the line is [-90, 90] degrees
            //and helps when comparing line angles to each other
            if (line[0] > line[2]) {
                lines.at(i) = { line[2], line[3], line[0], line[1] };
            }
            //precalculate line angles
            double angle = atan((line[3] - line[1]) / (line[2] - line[0] + 0.0001));

            angle = fabs(angle) == 0 ? 0.000001 : angle;

            lineAngles.push_back(angle);

            //add line to all grid squares the endpoint are inside
            int a = (int)line[1] / 256;
            int b = (int)line[0] / 256;
            int c = (int)line[3] / 256;
            int d = (int)line[2] / 256;

            lineGrid.at(a).at(b).push_back(i);

            if (a != c || b != d) lineGrid.at(c).at(d).push_back(i);
        }

        //thresholds
        /*const double angleThreshold = 0.2f;
        const double directMergeAngleThreshold = 0.1f;*/
        directMergeThreshold = directMergeThreshold * directMergeThreshold; // 18
        distThreshold = distThreshold * distThreshold; //8.5
        minLineLength = minLineLength * minLineLength; //12
        maxLineLength = maxLineLength * maxLineLength;

        //foreach grid square
        for (int grid_row_idx = 0; grid_row_idx < y; grid_row_idx++) {
            //update progressbar for each row
            //updateProgressBar(grid_row_idx, y * 2);
            for (int grid_col_idx = 0; grid_col_idx < x; grid_col_idx++) {

                //create a set (no duplicates) of all lines in or around the current grid square (3x3 grid)
                std::unordered_set<int> lineIndicies;

                for (int i = std::max(0, grid_row_idx - 1); i < std::min(y, grid_row_idx + 2); i++) {
                    for (int j = std::max(0, grid_col_idx - 1); j < std::min(x, grid_col_idx + 2); j++) {

                        for (auto x : lineGrid.at(i).at(j)) {
                            lineIndicies.insert(x);
                        }
                    }
                }

                //turn set into a vector for easy iteration
                std::vector<int> lineIndiciesVec(lineIndicies.begin(), lineIndicies.end());

                //current amount of lines in current grid square (will chance as lines are joined)
                int sizei = (int)lineGrid.at(grid_row_idx).at(grid_col_idx).size();

                //for each line in current grid square
                for (int i = 0; i < sizei; i++) {

                    //get first line idx
                    int idxi = lineGrid.at(grid_row_idx).at(grid_col_idx).at(i);

                    //get first line
                    auto& li = lines.at(idxi);

                    //if line is already joined
                    if (li[0] < 0) continue;

                    //get line angle
                    double liAngle = lineAngles.at(idxi);

                    //current amount of lines in set of all lines in and around current grid square (will chance as lines are joined)
                    int sizej = (int)lineIndiciesVec.size();

                    //compare line to all lines in and around current grid square
                    for (int j = 0; j < sizej; j++) {
                        //don't compare the same line against itself
                        int idxj = lineIndiciesVec.at(j);
                        if (idxi == idxj) continue;

                        //get second line
                        auto& lj = lines.at(idxj);

                        //if second line is already joined
                        if (lj[0] < 0) continue;

                        //get second line angle
                        const double ljAngle = lineAngles.at(idxj);

                        //calculate difference in angles
                        double slopeDelta = liAngle - ljAngle;

                        //make sure the difference is calculated correctly (for example a line with angle 89 degrees looks similar to -89 degrees)
                        slopeDelta = std::min(fabs(slopeDelta), fabs(slopeDelta - PI * liAngle / fabs(liAngle)));

                        //if the lines don't meet the angle threshold
                        if (slopeDelta > angleThreshold) continue;

                        //if the distance and angle between two conecting endpoint (first to last or last to first) 
                        //is within a distance and angle threshold.
                        //bool directMerge = false;

                        ////first to last if distance is less than threshold
                        //if(distSquared(li[0] - lj[2], li[1] - lj[3]) <= directMergeThreshold){

                        //    //get relative angle between end points
                        //    double relativePointsAngle = atan2f(lj[3] - li[1], lj[2] - li[0]);

                        //    //correct relative angle [-180, 180] -> [0, 360]
                        //    relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                        //    //correct precalculated line angle [-90, 90] - > [0, 360]
                        //    double directionLineAngle = PI + lineAngles.at(idxi);

                        //    //compare if angle is less than threshold
                        //    if(fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold){
                        //        directMerge = true;
                        //    }
                        //}
                        ////last to first if distance is less than threshold
                        //if (!directMerge && distSquared(lj[1] - li[3], lj[0] - li[2]) <= directMergeThreshold){

                        //    //get relative angle between end points
                        //    double relativePointsAngle = atan2f(lj[1] - li[3], lj[0] - li[2]);

                        //    //correct relative angle [-180, 180] -> [0, 360]
                        //    relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                        //    //correct precalculated line angle ([-90, 90] +/- 180) - > [0, 360]
                        //    double directionLineAngle = lineAngles.at(idxi) < 0 ? 2*PI + lineAngles.at(idxi) : lineAngles.at(idxi);

                        //    //compare if angle is less than threshold
                        //    if(fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold){
                        //        directMerge = true;
                        //    }
                        //}

                        //for readability
                        const double p1[2] = { lj[0], lj[1] };
                        const double p2[2] = { lj[2], lj[3] };
                        const double p3[2] = { li[0], li[1] };
                        const double p4[2] = { li[2], li[3] };

                        //if direct merge is true or the distance from the endpoint of one line to another line segment being 
                        //less than a threshold
                        if (distToSegmentSquared(li, p1) <= distThreshold ||
                            distToSegmentSquared(li, p2) <= distThreshold ||
                            distToSegmentSquared(lj, p3) <= distThreshold ||
                            distToSegmentSquared(lj, p4) <= distThreshold) {

                            //get the longest line pair
                            std::array<double, 4> LongestLine = longestLinesPointsPair(p1, p2, p3, p4);

                            //calculate the squared length of each lines
                            const double liLength = distSquared(li[2] - li[0], li[3] - li[1]);
                            const double ljLength = distSquared(lj[2] - lj[0], lj[3] - lj[1]);

                            //create a weighting based on the comparative lengths of the lines (basically weight longer lines as being quadratically more important)
                            const double liScale = liLength / (liLength + ljLength);
                            const double ljScale = ljLength / (liLength + ljLength);

                            //calculated the weighted average angle of the two lines
                            double averageAngle = fabs(liAngle - ljAngle) < fabs(liAngle - ljAngle - PI * liAngle / fabs(liAngle)) ? liAngle * liScale + ljAngle * ljScale : (liAngle - PI * liAngle / fabs(liAngle)) * liScale + ljAngle * ljScale;

                            //correct the average angle if it's not between [-pi/2, pi/2]
                            averageAngle = fabs(averageAngle) > PI / 2.0 ? averageAngle - PI * averageAngle / fabs(averageAngle) : averageAngle;

                            //calculate midpoint of longest lines points pair
                            const double mid[2] = { (LongestLine[2] + LongestLine[0]) / 2.0, (LongestLine[3] + LongestLine[1]) / 2.0 };

                            //get half the length of the longest lines points pair
                            const double newLineRadius = sqrt(distSquared(LongestLine[2] - LongestLine[0], LongestLine[3] - LongestLine[1])) / 2.0;

                            //create a new line of the same length as the longest lines points pair but rotated to be at the calculated average angle
                            const double rcos = newLineRadius * cos(averageAngle);
                            const double rsin = newLineRadius * sin(averageAngle);

                            std::array<double, 4> newLine = { std::max(0.0, mid[0] - rcos), std::min(y * 256.0 - 1.0, std::max(0.0, mid[1] - rsin)), std::min(x * 256.0 - 1.0, mid[0] + rcos), std::max(0.0 ,std::min(y * 256.0 - 1.0, mid[1] + rsin)) };

                            /*if (newLine[0] > newLine[2]) {
                                newLine = { newLine[2], newLine[3], newLine[0], newLine[1] };
                            }*/

                            //get new index
                            const int newIdx = (int)lines.size();
                            //add new line the list of lines
                            lines.push_back(newLine);
                            //calculate angle of new line
                            double angle = atan((newLine[3] - newLine[1]) / (newLine[2] - newLine[0] + 0.0001));

                            angle = fabs(angle) == 0 ? 0.000001 : angle;

                            lineAngles.push_back(angle);

                            //add line to grid
                            lineGrid.at(grid_row_idx).at(grid_col_idx).push_back(newIdx);
                            //increase for loop counter
                            sizei++;

                            //add line to set of lines in or around current grid square
                            lineIndiciesVec.push_back(newIdx);

                            //add line to any other grid squares it might also be in
                            int a = (int)newLine[1] / 256;
                            int b = (int)newLine[0] / 256;
                            int c = (int)newLine[3] / 256;
                            int d = (int)newLine[2] / 256;

                            if (a != grid_row_idx || b != grid_col_idx) {
                                lineGrid.at(a).at(b).push_back(newIdx);
                                if ((c != a && c != grid_row_idx) || (d != b && d != grid_col_idx)) {
                                    lineGrid.at(c).at(d).push_back(newIdx);
                                }
                            }
                            else if (c != grid_row_idx || d != grid_col_idx) {
                                lineGrid.at(c).at(d).push_back(newIdx);
                            }

                            //set x coord of first point of each now joined line to signify that the lines were joined
                            lines.at(idxi)[0] = -1.0;
                            lines.at(idxj)[0] = -1.0;
                            //no need to check the first line anymore
                            break;
                        }

                    }
                }

            }
        }

        //foreach grid square
        for (int grid_row_idx = 0; grid_row_idx < y; grid_row_idx++) {
            //update progressbar for each row
            //updateProgressBar(grid_row_idx + y, y * 2);
            for (int grid_col_idx = 0; grid_col_idx < x; grid_col_idx++) {

                //create a set (no duplicates) of all lines in or around the current grid square (3x3 grid)
                std::unordered_set<int> lineIndicies;

                for (int i = std::max(0, grid_row_idx - 1); i < std::min(y, grid_row_idx + 2); i++) {
                    for (int j = std::max(0, grid_col_idx - 1); j < std::min(x, grid_col_idx + 2); j++) {

                        for (auto x : lineGrid.at(i).at(j)) {
                            lineIndicies.insert(x);
                        }
                    }
                }

                //turn set into a vector for easy iteration
                std::vector<int> lineIndiciesVec(lineIndicies.begin(), lineIndicies.end());

                //current amount of lines in current grid square (will chance as lines are joined)
                int sizei = (int)lineGrid.at(grid_row_idx).at(grid_col_idx).size();

                //for each line in current grid square
                for (int i = 0; i < sizei; i++) {

                    //get first line idx
                    int idxi = lineGrid.at(grid_row_idx).at(grid_col_idx).at(i);

                    //get first line
                    auto& li = lines.at(idxi);

                    //if line is already joined
                    if (li[0] < 0) continue;

                    //get line angle
                    double liAngle = lineAngles.at(idxi);

                    //current amount of lines in set of all lines in and around current grid square (will chance as lines are joined)
                    int sizej = (int)lineIndiciesVec.size();

                    //compare line to all lines in and around current grid square
                    for (int j = 0; j < sizej; j++) {
                        //don't compare the same line against itself
                        int idxj = lineIndiciesVec.at(j);
                        if (idxi == idxj) continue;

                        //get second line
                        auto& lj = lines.at(idxj);

                        //if second line is already joined
                        if (lj[0] < 0) continue;

                        //get second line angle
                        const double ljAngle = lineAngles.at(idxj);

                        //calculate difference in angles
                        double slopeDelta = liAngle - ljAngle;

                        //make sure the difference is calculated correctly (for example a line with angle 89 degrees looks similar to -89 degrees)
                        slopeDelta = std::min(fabs(slopeDelta), fabs(slopeDelta - PI * liAngle / fabs(liAngle)));

                        //if the lines don't meet the angle threshold
                        if (slopeDelta > angleThreshold) continue;

                        //if the distance and angle between two connecting endpoint (first to last or last to first) 
                        //is within a distance and angle threshold.
                        bool directMerge = false;

                        //first to last if distance is less than threshold
                        if (distSquared(li[0] - lj[2], li[1] - lj[3]) <= directMergeThreshold) {

                            //get relative angle between end points
                            double relativePointsAngle = atan2(lj[3] - li[1], lj[2] - li[0]);

                            //correct relative angle [-180, 180] -> [0, 360]
                            relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                            //correct precalculated line angle [-90, 90] - > [0, 360]
                            double directionLineAngle = PI + lineAngles.at(idxi);

                            //compare if angle is less than threshold
                            if (fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold) {
                                directMerge = true;
                            }
                        }
                        //last to first if distance is less than threshold
                        if (!directMerge && distSquared(lj[1] - li[3], lj[0] - li[2]) <= directMergeThreshold) {

                            //get relative angle between end points
                            double relativePointsAngle = atan2(lj[1] - li[3], lj[0] - li[2]);

                            //correct relative angle [-180, 180] -> [0, 360]
                            relativePointsAngle = relativePointsAngle < 0 ? 2 * PI + relativePointsAngle : relativePointsAngle;

                            //correct precalculated line angle ([-90, 90] +/- 180) - > [0, 360]
                            double directionLineAngle = lineAngles.at(idxi) < 0 ? 2 * PI + lineAngles.at(idxi) : lineAngles.at(idxi);

                            //compare if angle is less than threshold
                            if (fabs(relativePointsAngle - directionLineAngle) <= directMergeAngleThreshold) {
                                directMerge = true;
                            }
                        }

                        //for readability
                        const double p1[2] = { lj[0], lj[1] };
                        const double p2[2] = { lj[2], lj[3] };
                        const double p3[2] = { li[0], li[1] };
                        const double p4[2] = { li[2], li[3] };

                        //if direct merge is true or the distance from the endpoint of one line to another line segment being 
                        //less than a threshold
                        if (directMerge ||
                            distToSegmentSquared(li, p1) <= distThreshold ||
                            distToSegmentSquared(li, p2) <= distThreshold ||
                            distToSegmentSquared(lj, p3) <= distThreshold ||
                            distToSegmentSquared(lj, p4) <= distThreshold) {

                            //get the longest line pair
                            std::array<double, 4> LongestLine = longestLinesPointsPair(p1, p2, p3, p4);

                            //calculate the squared length of each lines
                            const double liLength = distSquared(li[2] - li[0], li[3] - li[1]);
                            const double ljLength = distSquared(lj[2] - lj[0], lj[3] - lj[1]);

                            //create a weighting based on the comparative lengths of the lines (basically weight longer lines as being quadratically more important)
                            const double liScale = liLength / (liLength + ljLength);
                            const double ljScale = ljLength / (liLength + ljLength);

                            //calculated the weighted average angle of the two lines
                            double averageAngle = fabs(liAngle - ljAngle) < fabs(liAngle - ljAngle - PI * liAngle / fabs(liAngle)) ? liAngle * liScale + ljAngle * ljScale : (liAngle - PI * liAngle / fabs(liAngle)) * liScale + ljAngle * ljScale;

                            //correct the average angle if it's not between [-pi/2, pi/2]
                            averageAngle = fabs(averageAngle) > PI / 2.0 ? averageAngle - PI * averageAngle / fabs(averageAngle) : averageAngle;

                            //calculate midpoint of longest lines points pair
                            const double mid[2] = { (LongestLine[2] + LongestLine[0]) / 2.0, (LongestLine[3] + LongestLine[1]) / 2.0 };

                            //get half the length of the longest lines points pair
                            const double newLineRadius = sqrt(distSquared(LongestLine[2] - LongestLine[0], LongestLine[3] - LongestLine[1])) / 2.0;

                            //create a new line of the same length as the longest lines points pair but rotated to be at the calculated average angle
                            const double rcos = newLineRadius * cos(averageAngle);
                            const double rsin = newLineRadius * sin(averageAngle);

                            std::array<double, 4> newLine = { std::max(0.0, mid[0] - rcos), std::min(y * 256.0 - 1.0, std::max(0.0, mid[1] - rsin)), std::min(x * 256.0 - 1.0, mid[0] + rcos), std::max(0.0 ,std::min(y * 256.0 - 1.0, mid[1] + rsin)) };

                            /*if (newLine[0] > newLine[2]) {
                                newLine = { newLine[2], newLine[3], newLine[0], newLine[1] };
                            }*/

                            //get new index
                            const int newIdx = (int)lines.size();
                            //add new line the list of lines
                            lines.push_back(newLine);
                            //calculate angle of new line
                            double angle = atan((newLine[3] - newLine[1]) / (newLine[2] - newLine[0] + 0.0001));

                            angle = fabs(angle) == 0 ? 0.000001 : angle;

                            lineAngles.push_back(angle);

                            //add line to grid
                            lineGrid.at(grid_row_idx).at(grid_col_idx).push_back(newIdx);
                            //increase for loop counter
                            sizei++;

                            //add line to set of lines in or around current grid square
                            lineIndiciesVec.push_back(newIdx);

                            //add line to any other grid squares it might also be in
                            int a = (int)newLine[1] / 256;
                            int b = (int)newLine[0] / 256;
                            int c = (int)newLine[3] / 256;
                            int d = (int)newLine[2] / 256;

                            if (a != grid_row_idx || b != grid_col_idx) {
                                lineGrid.at(a).at(b).push_back(newIdx);
                                if ((c != a && c != grid_row_idx) || (d != b && d != grid_col_idx)) {
                                    lineGrid.at(c).at(d).push_back(newIdx);
                                }
                            }
                            else if (c != grid_row_idx || d != grid_col_idx) {
                                lineGrid.at(c).at(d).push_back(newIdx);
                            }

                            //set x coord of first point of each now joined line to signify that the lines were joined
                            lines.at(idxi)[0] = -1.0;
                            lines.at(idxj)[0] = -1.0;
                            //no need to check the first line anymore
                            break;
                        }

                    }
                }

            }
        }

        //adds new line after progress bar
        //std::cout << std::endl;

        std::vector<std::array<double, 4>> newLines;
        newLines.reserve((int)(lines.size() / 2));

        for (auto& l : lines) {
            if (l[0] >= 0) {
                const double d = distSquared(l[2] - l[0], l[3] - l[1]);

                if (d >= minLineLength && d <= maxLineLength) {
                    newLines.push_back(l);
                }
            }
        }

        return newLines;

        //int i = lines.size() - 1;

        ////loop through all lines and remove small lines / lines that were joined
        ////looping through backwards is faster under the circumstances
        //
        //while (i >= 0) {
        //    auto& l = lines.at(i);

        //    if (l[0] < 0 || distSquared(l[2] - l[0], l[3] - l[1]) < minLineLength) {
        //        lines.erase(lines.begin() + i);
        //    }
        //    i--;
        //}

        //return lines;
    }
};