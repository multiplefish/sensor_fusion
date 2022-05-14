#include "camera_camera_sync/camera_camera_sync.hpp"

int main()
{
    // CameraCameraSync camera_camera_sync_;
    CameraCameraSync *camera_camera_sync_=new CameraCameraStampSync();

    std::string oriDirs = "/root/workspace/data/practice_1_1_multi_camera_sync/camera_front_left_60";
    std::string dstDirs = "/root/workspace/data/practice_1_1_multi_camera_sync/camera_front_right_60";
    camera_camera_sync_->getImageTimeStamp(oriDirs, dstDirs);
    std::vector<std::pair<std::string, std::string> > syncImageLists;
    int number = camera_camera_sync_->getImageNumber();
    // 时间矫正
    if (number > 0)
    {
        syncImageLists = camera_camera_sync_->imageTimeStampSyncFuncion();
    }
    
    for(auto syncPair : syncImageLists)
    {
        cv::Mat image1 = cv::imread(syncPair.first, cv::IMREAD_GRAYSCALE);
        cv::Mat image2 = cv::imread(syncPair.second, cv::IMREAD_GRAYSCALE);
        if( !image1.data || !image2.data )
        { 
            std::cout<< " --(!) Error reading images " << std::endl; 
            return -1;
        }
        camera_camera_sync_->synchronizePitchRoll(image1,image2);
        // camera_camera_sync_->spatialSynchronization(image1, image2);
    }



}
