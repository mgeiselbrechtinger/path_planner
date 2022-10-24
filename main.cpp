#include <math.h>

#include <iostream>
#include <vector>
#include <limits>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PathPlanner
{
    private:
        Mat map;
        Point2i start;
        vector<Point2i> path;
        const uchar DRIVABLE = 255;
        const uchar DRIVABLE_THRESH = 250;
        float obst_avoid_weight = 10.0;

    public:

        PathPlanner();

        PathPlanner(char *fname)
        {
            load_map(fname);
        }
        
        void load_map(char *fname)
        {
            map = imread(fname, IMREAD_GRAYSCALE);
            transpose(map, map);
        }

        void set_start_point(int x, int y)
        {
            start = Point2i(x, y);
        }

        void set_obst_avoid_weight(float alpha)
        {
            obst_avoid_weight = alpha;
        }

        void show_map()
        {
            Mat disp_map = map.clone();

            // draw path
            for(auto p : path)
                disp_map.at<uchar>(p.y, p.x) = 100;

            // draw car
            rectangle(disp_map,
                    Point(start.x-2, start.y), 
                    Point(start.x+2, start.y+8), 
                    0, FILLED);

            imshow("Map", disp_map);
            waitKey(0);
        }

        bool find_path()
        {
            Point2i goal(start.x, start.y+2);

            // prepare maps
            Mat search_map, obst_dist_map; 
            threshold(map, search_map, DRIVABLE_THRESH, DRIVABLE, THRESH_BINARY);
            distanceTransform(search_map, obst_dist_map, DIST_L2, 5);

            // block going backwards
            const float inf = numeric_limits<float>::infinity();
            int y = start.y + 1;
            // check left
            for(int x = start.x; x >= 0; x--) {
                if(search_map.at<uchar>(y, x) == 0) 
                    break;
                else
                    search_map.at<uchar>(y, x) = 0;
            }
            // check right
            for(int x = start.x + 1; x < search_map.cols; x++) {
                if(search_map.at<uchar>(y, x) == 0) 
                    break;
                else
                    search_map.at<uchar>(y, x) = 0;
            }

            // plain Dijkstra
            
            // initialization
            typedef struct {
                Point2i coord;
                float dist;
                int pred;
                bool erased;
            } node_t;
            // TODO use priority queue instead
            vector<node_t> candidates;

            // Use flood fill variant if track only small portion of map
            // or even use constrcutive variant where nodes are pushed 
            // on the go during search
            for(int x = 0; x < search_map.rows; x++) {
                for(int y = 0; y < search_map.cols; y++) {

                    if(search_map.at<uchar>(y, x) == DRIVABLE) {
                        node_t c = {Point2i(x, y), inf, -1, false};

                        if(c.coord == start) 
                            c.dist = 0;

                        candidates.push_back(c);
                    }
                }
            }

            // path search
            int cur_idx;
            while(!candidates.empty()) {
                // find candidate with min dist
                float min_dist = inf;
                for(size_t i = 0; i < candidates.size(); i++) {
                    node_t &c = candidates[i];
                    if(c.erased) continue;
                    if(c.dist < min_dist) {
                        min_dist = c.dist;
                        cur_idx = i;
                    }
                }

                node_t &cur_c = candidates[cur_idx];
                cur_c.erased = true;

                if(cur_c.coord == goal)
                    break;

                // iterate over neighbors
                for(size_t i = 0; i < candidates.size(); i++) {
                    node_t &c = candidates[i];
                    if(c.erased) 
                        continue;
                    float alt_dist = cur_c.dist;
                    float dx = abs(c.coord.x - cur_c.coord.x);
                    float dy = abs(c.coord.y - cur_c.coord.y);
                    // diagonal neighbor
                    if(dx == 1 && dy == 1)
                        alt_dist += 1.414;
                    // direct neighbor
                    else if(dx <= 1 && dy <= 1)
                        alt_dist += 1;
                    // no neighbor
                    else
                        continue;
                    // heuristic weight to avoid obstacles
                    alt_dist += obst_avoid_weight/obst_dist_map.at<float>(c.coord);
                    // update with alternative route
                    if(alt_dist < c.dist) {
                        c.dist = alt_dist;
                        c.pred = cur_idx;
                    }
                }
            }

            // construct reversed path: goal -> start
            vector<Point2i> rpath;
            node_t &c = candidates[cur_idx];
            rpath.push_back(c.coord);
            while(c.pred != -1) {
                c = candidates[c.pred];
                rpath.push_back(c.coord);
            }

            // copy and reverse path: start -> goal
            path = vector<Point2i>(rpath.rbegin(), rpath.rend());
            
            return true;
        }

};

int main(int argc, char *argv[])
{
    if(argc < 2) {
        cout << "Missing file name" << endl;
        return -1;
    }

    PathPlanner planner(argv[1]);
    planner.set_start_point(220, 220); // TODO read from yaml file
    planner.find_path();
    planner.show_map();
    
    return 0;
}
