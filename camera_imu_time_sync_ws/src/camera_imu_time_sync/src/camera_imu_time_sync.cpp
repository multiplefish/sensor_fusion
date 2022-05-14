#include "camera_imu_time_sync.hpp"
// #include "camera_imu_time_sync/utils.hpp"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int arg,char** argv){

    LOG(INFO) << "hello world"<<endl;
    ros::init(arg,argv,"camera_imu_time_sync");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    return 0;
}