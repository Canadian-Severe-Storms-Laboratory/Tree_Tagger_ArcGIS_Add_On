import math
import shapefile
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import numpy as np

if __name__ == '__main__':

    auto_directions = []

    with shapefile.Reader("shapefiles/Auto_median_v2.shp") as sfr:
        for shape in sfr.shapes():
            auto_directions.append([])

            p = shape.points
            auto_directions[len(auto_directions) - 1].append((p[0][0] + p[1][0]) / 2.0)
            auto_directions[len(auto_directions) - 1].append((p[0][1] + p[1][1]) / 2.0)

            angle = math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
            angle = (2*math.pi + angle) if angle < 0 else angle

            angle = math.degrees(angle)

            auto_directions[len(auto_directions) - 1].append(angle)

    #print(auto_directions)

    manual_directions = []

    with shapefile.Reader("shapefiles/Autotreefall_tornadic_corrected_v3") as sfr:
        #print(sfr.fields)

        for shape in sfr.shapes():
            manual_directions.append(shape.points[0])

        for i in range(len(sfr.records())):
            manual_directions[i].append(sfr.record(i)[4])

    #print(manual_directions)

    bins = [0 for i in range(36)]
    deltas = []
    dists = []

    i = 0
    while i < len(auto_directions) - 1:

        ad = auto_directions[i]
        ad2 = auto_directions[i+1]
        t = False

        if abs(ad[0] - ad2[0]) < 1 and abs(ad[1] - ad2[1]) < 1:
            t = True
            i += 1

        shortest_distance = 1e12
        delta_angle = 180
        #x = manual_directions[0]

        for md in manual_directions:
            distance = math.hypot(ad[0] - md[0], ad[1] - md[1])

            if distance < shortest_distance:
                #x = md
                shortest_distance = distance
                if t:
                    delta_angle = abs(ad[2] - md[2])
                    delta_angle = min(delta_angle, 360 - delta_angle)
                    delta_angle2 = abs(ad2[2] - md[2])
                    delta_angle2 = min(delta_angle2, 360 - delta_angle2)
                    delta_angle = min(delta_angle, delta_angle2)

                else:
                    delta_angle = abs(ad[2] - md[2])
                    delta_angle = min(delta_angle, 360 - delta_angle)

        #print(ad, x, delta_angle)
        dists.append(shortest_distance)
        deltas.append(delta_angle)
        bins[int(math.floor(delta_angle / 5))] += 1
        i += 1

    deltas.sort()
    # print(dists)
    # print(statistics.mean(dists), statistics.median(dists))
    print(statistics.mean(deltas), statistics.median(deltas), deltas[int(len(deltas)*0.8 - 1)])
    print(bins)
    print(sum(bins[0:6]), sum(bins[6:]))

    plt.hist(deltas, bins=36, weights=np.ones(len(deltas)) / len(deltas))
    plt.title('Histogram of Angle Differences (Median)')
    plt.xlabel('Angle Difference (Degrees)')
    plt.ylabel('Percentage of Vectors')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.axvline(statistics.mean(deltas), color='k', linestyle='dashed', linewidth=1, label="Mean: " + str(round(statistics.mean(deltas), 1)))
    plt.axvline(statistics.median(deltas), color='k', linestyle='solid', linewidth=1, label="Median: " + str(round(statistics.median(deltas), 1)))
    plt.axvline(deltas[int(len(deltas)*0.8 - 1)], color='k', linestyle='dotted', linewidth=1, label="80%: " + str(round(deltas[int(len(deltas)*0.8 - 1)], 1)))
    plt.legend()
    plt.show()