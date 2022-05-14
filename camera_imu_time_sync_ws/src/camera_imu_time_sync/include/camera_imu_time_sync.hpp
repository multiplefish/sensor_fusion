#pragma once
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <memory>
#include <Eigen/Core>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cdkf.h>
class CameraImuTimeSync{

    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraImuTimeSync(const ros::NodeHandle&nh,const ros::NodeHandle&nh_private):
    nh_(nh),nh_private_(nh_private),it_(nh_private_),stamp_on_arrival_(false),max_imu_data_age_s_(2.0),
    delay_by_n_frames_(5),focal_length_(460),calc_offset_(true){

        nh_private_.param("stamp_on_arrival", stamp_on_arrival_, stamp_on_arrival_);
        nh_private_.param("max_imu_data_age_s", max_imu_data_age_s_, max_imu_data_age_s_);
        nh_private_.param("delay_by_n_frames", delay_by_n_frames_, delay_by_n_frames_);
        nh_private_.param("focal_length", focal_length_, focal_length_);
        nh_private_.param("calc_offset", calc_offset_, calc_offset_);
        setupCDKF();
        constexpr int kImageQueueSize = 10;
        constexpr int kImuQueueSize = 100;
        constexpr int kFloatQueueSize = 100;
        

    
    };
    void setupCDKF(){
    CDKF::Config config;

    nh_private_.param("verbose", config.verbose, config.verbose);
    nh_private_.param("mah_threshold", config.mah_threshold, config.mah_threshold);

    nh_private_.param("inital_delta_t", config.inital_delta_t, config.inital_delta_t);
    nh_private_.param("inital_offset", config.inital_offset, config.inital_offset);

    nh_private_.param("inital_delta_t_sd", config.inital_delta_t_sd, config.inital_delta_t_sd);
    nh_private_.param("inital_offset_sd", config.inital_offset_sd, config.inital_offset_sd);

    nh_private_.param("timestamp_sd", config.timestamp_sd, config.timestamp_sd);

    nh_private_.param("delta_t_sd", config.delta_t_sd, config.delta_t_sd);
    nh_private_.param("offset_sd", config.offset_sd, config.offset_sd);

    cdkf_ = std::unique_ptr<CDKF>(new CDKF(config));
    };


    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    image_transport::ImageTransport it_;

    ros::Subscriber imu_sub_;
    image_transport::Subscriber image_sub_;

    ros::Publisher delta_t_pub_;
    ros::Publisher offset_pub_;
    image_transport::Publisher image_pub_;

    // 是否初始帧
    bool stamp_on_arrival_;
    // 最大的imu
    double max_imu_data_age_s_;
    // 窗口的大小
    int delay_by_n_frames_;
    // 焦距
    double focal_length_;
    //时间补偿的机制
    bool calc_offset_;

    std::unique_ptr<CDKF> cdkf_;

    IMUList imu_rotations_;

};