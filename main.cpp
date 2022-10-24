#include <math.h>
#include <unistd.h>

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
        float obst_avoid_weight = 1.0;

        typedef struct {
            Point2i coord;
            float dist;
            int pred;
            bool erased; // lazy deletion
        } node_t;

        static bool node_cmp(const node_t &a, const node_t &b)
        {
            if(a.erased)
                return false;
            else
                return a.dist < b.dist;
        }

        static bool never_seen(const vector<node_t> &vec, const Point2i &pt)
        {
            for(auto &e : vec) {
                if(e.coord == pt)
                    return false;
            }
            return true;
        }

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
            priority_queue<node_t, vector<node_t>, decltype(&node_cmp)> candidates(&node_cmp);
            vector<node_t> visited;

            // path search
            candidates.push({start, 0, -1, false});
            int cur_idx;
            while(!candidates.empty()) {
                node_t cur_c;
                // find candidate with min dist
                do {
                    cur_c = candidates.top();
                    candidates.pop();
                } while(cur_c.erased && !candidates.empty());
                cur_idx = visited.size();
                visited.push_back(cur_c);

                cout << cur_c.coord << endl;
                if(cur_c.coord == goal)
                    break;

                // iterate over neighbors
                float alt_dist = cur_c.dist;
                // up
                Point2i pt_u(cur_c.coord.x, cur_c.coord.y - 1);
                if(pt_u.y >= 0 
                        && search_map.at<uchar>(pt_u) == DRIVABLE
                        && never_seen(visited, pt_u)) {
                    candidates.push({pt_u, alt_dist + 1, cur_idx, false});
                }
                // down
                Point2i pt_d(cur_c.coord.x, cur_c.coord.y + 1);
                if(pt_d.y < search_map.cols 
                        && search_map.at<uchar>(pt_d) == DRIVABLE
                        && never_seen(visited, pt_d)) {
                    candidates.push({pt_d, alt_dist + 1, cur_idx, false});
                }
                // left
                Point2i pt_l(cur_c.coord.x - 1, cur_c.coord.y);
                if(pt_l.x >= 0 
                        && search_map.at<uchar>(pt_l) == DRIVABLE
                        && never_seen(visited, pt_l)) {
                    candidates.push({pt_l, alt_dist + 1, cur_idx, false});
                }
                // right
                Point2i pt_r(cur_c.coord.x + 1, cur_c.coord.y);
                if(pt_r.x < search_map.rows 
                        && search_map.at<uchar>(pt_r) == DRIVABLE
                        && never_seen(visited, pt_r)) {
                    candidates.push({pt_r, alt_dist + 1, cur_idx, false});
                }
                // left-up
                Point2i pt_lu(cur_c.coord.x - 1, cur_c.coord.y - 1);
                if(pt_lu.x >= 0 && pt_lu.y >= 0 
                        && search_map.at<uchar>(pt_lu) == DRIVABLE
                        && never_seen(visited, pt_lu)) {
                    candidates.push({pt_lu, alt_dist + 1.414, cur_idx, false});
                }
                // right-up
                Point2i pt_ru(cur_c.coord.x + 1, cur_c.coord.y - 1);
                if(pt_ru.x < search_map.rows && pt_ru.y >= 0 
                        && search_map.at<uchar>(pt_ru) == DRIVABLE
                        && never_seen(visited, pt_ru)) {
                    candidates.push({pt_ru, alt_dist + 1.414, cur_idx, false});
                }
                // left-down
                Point2i pt_ld(cur_c.coord.x - 1, cur_c.coord.y + 1);
                if(pt_ld.x >= 0 && pt_ld.y < search_map.cols 
                        && search_map.at<uchar>(pt_ld) == DRIVABLE
                        && never_seen(visited, pt_ld)) {
                    candidates.push({pt_ld, alt_dist + 1.414, cur_idx, false});
                }
                // right-down
                Point2i pt_rd(cur_c.coord.x + 1, cur_c.coord.y + 1);
                if(pt_rd.x < search_map.rows && pt_rd.y < search_map.cols 
                        && search_map.at<uchar>(pt_rd) == DRIVABLE
                        && never_seen(visited, pt_rd)) {
                    candidates.push({pt_rd, alt_dist + 1.414, cur_idx, false});
                }
            }

            // construct reversed path: goal -> start
            vector<Point2i> rpath;
            node_t &c = visited[cur_idx];
            rpath.push_back(c.coord);
            while(c.pred != -1) {
                c = visited[c.pred];
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
