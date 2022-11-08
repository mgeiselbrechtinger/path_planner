#include <unistd.h>

#include <cmath>
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
        bool map_available = false;

        Point2i start;
        bool start_point_available;

        vector<Point2i> path;
        bool path_available;

        Mat map_driveable;
        enum driveable_t {
            not_driveable=0, 
            driveable=255, 
            driveable_threshold=250
        };

        float obst_avoid_weight = 1.0;

    public:

        PathPlanner();

        PathPlanner(const string &fname)
        {
            load_map(fname);
        }
        
        void load_map(const string &fname)
        {
            map = imread(fname, IMREAD_GRAYSCALE);
            transpose(map, map);
            threshold(map, map_driveable, driveable_threshold, driveable, THRESH_BINARY);
            map_available = true;
        }

        void set_start_point(int x, int y)
        {
            start = Point2i(x, y);
            start_point_available = true;
        }

        void set_obst_avoid_weight(float alpha)
        {
            obst_avoid_weight = alpha;
        }

        void show_map(bool full=true)
        {
            if(!map_available) {
                cerr << "No map loaded to print" << endl;
                return;
            }
            
            Mat disp_map;
            if(full) {
                disp_map = map.clone();

                // draw path
                if(path_available) {
                    for(auto p : path)
                        disp_map.at<uchar>(p.y, p.x) = 100;
                }

                // draw car
                if(start_point_available) {
                    rectangle(disp_map,
                            Point(start.x-2, start.y), 
                            Point(start.x+2, start.y+8), 
                            0, FILLED);
                }

            } else {
                disp_map = map_driveable;

            }

            imshow("Map", disp_map);
            waitKey(0);
            destroyWindow("Map");
        }

        bool find_path()
        {
            if(!map_available || !start_point_available)
                return false;

            constexpr float inf = numeric_limits<float>::infinity();
            constexpr float sqrt2 = sqrt(2);

            // prepare maps
            Mat search_map, obst_dist_map; 
            search_map = map_driveable.clone();
            distanceTransform(search_map, obst_dist_map, DIST_L2, 5);

            Point2i goal(start.x, start.y+2);
            // block going backwards
            int y = start.y + 1;
            // check left
            for(int x = start.x; x >= 0; x--) {
                if(search_map.at<uchar>(y, x) == not_driveable) 
                    break;
                else
                    search_map.at<uchar>(y, x) = not_driveable;
            }
            // check right
            for(int x = start.x + 1; x < search_map.cols; x++) {
                if(search_map.at<uchar>(y, x) == not_driveable) 
                    break;
                else
                    search_map.at<uchar>(y, x) = not_driveable;
            }

            // Dijkstra with wall-distance heuristic
            
            struct node_t {
                Point2i coord;
                float dist;
                int pred_idx;
                bool erased; // lazy deletion
            };
            vector<node_t> candidates;

            // Use flood fill variant if track only small portion of map
            // or even use constrcutive variant where nodes are pushed 
            // on the go during search
            for(int x = 0; x < search_map.rows; x++) {
                for(int y = 0; y < search_map.cols; y++) {

                    if(search_map.at<uchar>(y, x) == driveable) {
                        node_t c {Point2i(x, y), inf, -1, false};

                        if(c.coord == start) 
                            c.dist = 0;

                        candidates.push_back(c);
                    }
                }
            }

            // Path search
            int cur_idx;
            while(!candidates.empty()) {
                // find candidate with min dist
                float min_dist = inf;
                for(size_t i = 0; i < candidates.size(); i++) {
                    const node_t &c = candidates[i];

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

                    if(c.erased) continue;

                    float alt_dist = cur_c.dist;
                    float dx = abs(c.coord.x - cur_c.coord.x);
                    float dy = abs(c.coord.y - cur_c.coord.y);
                    // diagonal neighbor
                    if(dx == 1 && dy == 1)
                        alt_dist += sqrt2;
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
                        c.pred_idx = cur_idx;
                    }
                }
            }

            // construct reversed path: goal -> start
            vector<Point2i> rpath;
            node_t &c = candidates[cur_idx];
            rpath.push_back(c.coord);
            while(c.pred_idx != -1) {
                c = candidates[c.pred_idx];
                rpath.push_back(c.coord);
            }

            // copy and reverse path: start -> goal
            path.clear();
            path = vector<Point2i>(rpath.rbegin(), rpath.rend());
            path_available = true;
            
            return true;
        }

};

int main(int argc, char *argv[])
{
    if(argc < 2) {
        cout << "Missing file name" << endl;
        return -1;
    }

    char *fname = argv[1];

    int opt;
    float obst_avoid_weight = 0.0;
    while((opt = getopt(argc, argv, "a:")) != -1) {
        switch(opt) {
            case 'a':
                obst_avoid_weight = stof(optarg);
                break;
            
            case '?':
                cerr << "Usage: " << argv[0] << " path/to/map [-a obst-avoid-weight]" << endl;
                return -1;
        }
    }

    PathPlanner planner(fname);
    planner.set_start_point(220, 220); // TODO read from yaml file
    planner.set_obst_avoid_weight(obst_avoid_weight);
    planner.find_path();
    planner.show_map();
    
    return 0;
}
