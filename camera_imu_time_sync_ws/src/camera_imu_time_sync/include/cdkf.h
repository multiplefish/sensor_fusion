#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <Eigen/Eigen>
#include <iostream>
#include <list>


template <class Type>
    using AlignedList =std::list<Type,Eigen::aligned_allocator<Type>>;
    using IMUList =AlignedList<std::pair<ros::Time,Eigen::Quaterniond> >;

class CDKF{

    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct Config
    {
        bool verbose=true;
        double mah_threshold=10.0;
        // 初始值状态值
        double inital_delta_t=0.05;
        double inital_offset=0.0;
        // 初始化状态噪声
        double inital_timestamp_sd=0.1;
        double inital_delta_t_sd=0.1;
        double inital_offset_sd=0.1;
        // 初始化测量噪声
        double timestamp_sd=0.02;
        double angular_velocity_sd=0.03;

        // 传播噪声
        double delta_t_sd=0.0001;
        double offset_sd = 0.0001;
    };
    CDKF(const Config &config);


};