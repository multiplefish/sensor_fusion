#ifndef _CAMERA_CAMERA_SYNC_HPP_
#define _CAMERA_CAMERA_SYNC_HPP_

#include <string>
#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"
#include <glog/logging.h>
#include <time.h>

#define PRINTLOG std::cout << __LINE__ << " in " << __FILE__ << std::endl;


using namespace std;
using namespace cv;


#define PI 3.14159265
const double UTILS_MATCH_MIN_DIST = 25.0;
const double UTILS_FEATURE_MATCH_RATIO_THRESH = 0.5;
struct CamParams
{
    // left: cam1, right: cam1
    Mat K1, K2, D1, D2, R, T;
    Size imgSize;

    CamParams()
    {
        // camera1:intrinsic
        double arrK1[9] = {1.9457000000000000e+03, 0., 8.9667539999999997e+02, 0.,1.9433000000000000e+03, 5.0516239999999999e+02, 0., 0., 1.};
        // camera2:intrinsic
        double arrK2[9] = {1.9492000000000000e+03, 0., 9.2153930000000003e+02, 0.,1.9472000000000000e+03, 5.6912049999999999e+02, 0., 0., 1.};

        double arrR[9] = {1.0,0.,0.,0.,1.0,0.,0.,0.,1.0};
        double arrT[3] = {-1.1,0,0};
        // camera1:distortion coefficient
        double arrD1[5] = {5.7289999999999996e-01, 3.10500e-01,3.50001e-03, 6.8995000005e-04,-3.3500000000000002e-02};
        // camera2:distortion coefficient
        double arrD2[5] = {-5.8879999999999999e-01, 3.0020000000000002e-01,2.0999999999999999e-03, -2.2568999999999999e-04,2.1190000000000001e-01};

        K1 = Mat(3,3, CV_64F, arrK1).clone();
        K2 = Mat(3,3, CV_64F, arrK2).clone();

        R = Mat(3,3, CV_64F,arrR).clone();
        T = Mat(3,1, CV_64F,arrT).clone();

        D1 = Mat(1,5,CV_64F,arrD1).clone();
        D2 = Mat(1,5,CV_64F,arrD2).clone();

        imgSize.width = 1920;
        imgSize.height = 1080;
    }

    ~CamParams(){}

};
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {InputsAtCompileTime = NX, ValuesAtCompileTime = NY};
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs;}
    int values() const { return m_values;}
};

class MeanFunctor : public Functor<double> 
{
public:
    MeanFunctor(vector<vector<Point2f> > data):
        Functor<double>(2, data[0].size()), // 1:number of parameters to optimize
        _data(data),
        _cam(CamParams()) {}


    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
        // pitch
        double arrDetlaRPitch[9] = {1., 0., 0.,
                                    0.,cos(x[0]),-sin(x[0]),
                                    0.,sin(x[0]),cos(x[0])};
        Mat detlaRPitch = Mat(3,3,CV_64F, arrDetlaRPitch);
        // roll
        double arrDetlaRRoll[9] = {cos(x[1]),-sin(x[1]), 0.,
                                    sin(x[1]),cos(x[1]),0.,
                                    0.,0.,1.};
        Mat detlaRRoll = Mat(3,3,CV_64F, arrDetlaRRoll);
        
        // add disturb
        double distPitch = 3.14159265/180 * 2;
        double arrDistPitch[9] = {1.,0.,0.,
                                    0.,cos(distPitch), -sin(distPitch),
                                    0.,sin(distPitch), cos(distPitch)};
        
        Mat distRPitch = Mat(3,3,CV_64F, arrDistPitch);

        // 
        Mat optimR = _cam.R * detlaRPitch * detlaRRoll * distRPitch;
        //stereo rectify
        Mat R1, R2, P1, P2, Q;
        stereoRectify(_cam.K1, _cam.D1, _cam.K2, _cam.D2, _cam.imgSize, optimR, _cam.T, R1, R2, P1, P2, Q);

        //  points rectify
        vector<Point2f> rectpts1, rectpts2;
        undistortPoints(_data[0], rectpts1, _cam.K1, _cam.D1, R1, P1);
        undistortPoints(_data[1], rectpts2, _cam.K2, _cam.D2, R2, P2);

        // cost :L1/L2
        for(size_t i =0; i< rectpts1.size(); i++)
        {
            fvec[i] = abs(rectpts1[i].y - rectpts2[i].y);
        }

        return 0;
    }

private:
    vector<vector<Point2f> > _data;
    CamParams _cam;
};



class CameraCameraSync
{
public:
    CameraCameraSync(){};
    virtual ~CameraCameraSync(){};
    // 说明：获取图像的时间戳，这里的时间戳就是文件名，所以可以理解为直接获取文件名
    // 然后将获取的文件列表保存在两个队列中，方便后续找到对应的时间戳
    void getImageTimeStamp(std::string oriDirName, std::string dstDirName);

    int getImageNumber();

    // 说明：返回两个图像时间最接近的图像
    std::vector<std::pair<std::string, std::string> > imageTimeStampSyncFuncion();

    // 说明：评估两个图像的时间是否最接近的方法
    // 假设已经完成了时间硬件同步且两者曝光时间、帧率相同，内参一致，那么两个相机帧之间不会相差太多,
    // 如果完全同步，则两者的图像距离最接近，所以采用距离信息进行评价.
    // 假设 队列A中的元素n应该与队列B中的元素n距离最近，仅仅与B中的元素n-1，n+1进行比较，如果相差太多，那么认为时间硬件有问题
    virtual double evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage)=0;
    // double evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage){return 0;};
    // 空间同步
    virtual void  spatialSynchronization(cv::Mat srcImage1, cv::Mat srcImage2)=0;
    virtual bool synchronizePitchRoll(cv::Mat img_left, cv::Mat img_right)=0;


// private:
    std::vector<std::string> oriImageLists_;
    std::vector<std::string> dstImageLists_;
    std::vector<double> pitch_cache_;
    std::vector<double> roll_cache_;

    float timeThreshold_;

    void getFiles(std::string path, std::vector<std::string>& files);
    double getbaseTime(std::string pngfilenames, std::string patt);
};
class CameraCameraStampSync:public CameraCameraSync{
    public:
    CameraCameraStampSync(){};
    virtual ~CameraCameraStampSync(){};

    public:
    virtual double evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage){

        float meanX=0;
        float meanY=0;


        
        // float c1=std::pow(0.01*std::min(orgImage.rows,orgImage.cols),2);
        // float c2=std::pow(0.03*std::min(orgImage.rows,orgImage.cols),2);
        // float c3=c2/2;
        double c1 = 6.5025;
        double c2 = 58.5225;
        for(int i=0;i<orgImage.rows;++i){
            for(int j=0;j<orgImage.cols;++j){
                meanX+=orgImage.at<uchar>(i,j);
                meanY+=dstImage.at<uchar>(i,j);
            }
        }
        meanX=meanX/(orgImage.rows*orgImage.cols);
        meanY=meanY/(orgImage.rows*orgImage.cols);
        
        float sigmaX=0;
        float sigmaY=0;
        float sigmaXY=0;

        for(int i=0;i<orgImage.rows;++i){
            for(int j=0;j<orgImage.cols;++j){
                sigmaX+=((orgImage.at<uchar>(i,j)-meanX)*(orgImage.at<uchar>(i,j)-meanX));
                sigmaY+=((dstImage.at<uchar>(i,j)-meanY)*(dstImage.at<uchar>(i,j)-meanY));
                sigmaXY+=std::abs(((orgImage.at<uchar>(i,j)-meanX)*(dstImage.at<uchar>(i,j)-meanY)));
            }
        }
        sigmaX=sigmaX/((orgImage.rows)*(orgImage.cols)-1);
        sigmaY=sigmaY/((dstImage.rows)*(dstImage.cols)-1);
        sigmaXY=sigmaXY/((dstImage.rows)*(dstImage.cols)-1);

        // float l_x_y=L_x_y(meanX,meanY,c1);
        // float c_x_y=L_x_y(sigmaX,sigmaY,c2);
        // float s_x_y=(sigmaXY+c3)/(sigmaX*sigmaY+c3);
        double molecule = (2 * meanX*meanY + c1) * (2 * sigmaXY + c2);
        double denominator = (meanY*meanX + meanY * meanY + c1) * (sigmaX + sigmaY + c2);
        double ssim = molecule / denominator;

        
        // float ssim=l_x_y*c_x_y*s_x_y;
        return ssim;

    };
    inline float L_x_y(float x,float y,float c){
        float mole=2*x*y+c;
        float den=x*x+y*y+c;
        return mole/den;
    };
    virtual void spatialSynchronization(cv::Mat srcImage1, cv::Mat srcImage2){
         // 提取特征点    
//     //使用SURF算子检测关键点
	int minHessian = 400;//SURF算法中的hessian阈值
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;//vector模板类，存放任意类型的动态数组
    cv::Mat descriptors_object, descriptors_scene;
	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create(minHessian);
	
    cv::Ptr <cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create(minHessian);
    
	//调用detect函数检测出SURF特征关键点，保存在vector容器中
    // 特征点
	detector->detect(srcImage1, keypoints_object);
	detector->detect(srcImage2, keypoints_scene);

    //特征点描述，为下边的特征点匹配做准备  
    // 描述子
    cv::Mat matshow1, matshow2, kp1, kp2; 
    extractor->compute(srcImage1, keypoints_object, descriptors_object);
	extractor->compute(srcImage2, keypoints_scene, descriptors_scene);

    //使用FLANN匹配算子进行匹配
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matchePoints;
    matcher.match(descriptors_object, descriptors_scene, matchePoints);

    //最小距离和最大距离
    double max_dist = 0; 
    double min_dist = 100;

	//计算出关键点之间距离的最大值和最小值
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matchePoints[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf(">Max dist 最大距离 : %f \n", max_dist);
	printf(">Min dist 最小距离 : %f \n", min_dist);

	//匹配距离小于3*min_dist的点对
	std::vector< cv::DMatch > goodMatches;
 
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matchePoints[i].distance < 2.5 * min_dist)
		{
			goodMatches.push_back(matchePoints[i]);
		}
	}
    
	// //绘制出匹配到的关键点
	// cv::Mat imgMatches;
	// cv::drawMatches(srcImage1, keypoints_object, srcImage2, keypoints_scene,
	// 	goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
	// 	std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 
	//定义两个局部变量
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
 
	//从匹配成功的匹配对中获取关键点
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keypoints_object[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodMatches[i].trainIdx].pt);
	}
 
	cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);//计算透视变换 
 
	//从待测图片中获取角点
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cv::Point(0, 0);
	obj_corners[1] = cv::Point(srcImage1.cols, 0);
	obj_corners[2] = cv::Point(srcImage1.cols, srcImage1.rows);
	obj_corners[3] = cv::Point(0, srcImage1.rows);
	std::vector<cv::Point2f> scene_corners(4);
 
	//进行透视变换
	cv::perspectiveTransform(obj_corners, scene_corners, H);
 
	// //显示最终结果
	// imshow("效果图", imgMatches);
    // cv::waitKey(30);
    time_t timep;
    time(&timep);
    
    char name[1024];
    sprintf(name, "效果_%d.jpg", timep);
    
    // cv::imwrite(name,imgMatches);
    };
    bool synchronizePitchRoll(cv::Mat img_left, cv::Mat img_right){
    //     if(!imgLeft.data||!imgRight.data) {
    //         LOG(ERROR) << "no image data!";
    //         return false;
    //     }
    // std::vector<cv::Point2f>left_pts,right_pts;
    // findMatchPoints(imgLeft, imgRight, left_pts, right_pts);
    // LOG(INFO)<<"find match points:size: left:"<<left_pts.size() << " right: " << right_pts.size() << std::endl;

    // // LM 算法解决roll pitch
    // vector<vector<cv::Point2f> > data = {left_pts, right_pts};
    // Eigen::VectorXd x(2);
    // x<<0.,0.;
    // MeanFunctor functor(data);
    // Eigen::NumericalDiff<MeanFunctor> num_diff(functor, 1e-6);
    // Eigen::LevenbergMarquardt<Eigen::NumericalDiff<MeanFunctor>, double> lm(num_diff);
    // int info = lm.minimize(x);

    // std::cout << "current result: pitch & roll: " << x[0]/PI*180 << " " << x[1]/PI*180 << endl;

    // pitch_cache_.push_back(x[0]);
    // roll_cache_.push_back(x[1]);
    // return true;

    if(!img_left.data || !img_right.data )
    {
        ROS_ERROR_STREAM("no image data!");
        return false;
    }

    std::vector<cv::Point2f> left_pts, right_pts;
    findMatchPoints(img_left, img_right, left_pts, right_pts);
    std::cout << "find match points:size: left:" << left_pts.size() << " right: " << right_pts.size() << std::endl;

    // solve pitch and roll between cameras
    vector<vector<Point2f> > data = {left_pts, right_pts};
    Eigen::VectorXd x(2);
    x << 0., 0.;

    MeanFunctor functor(data);
    Eigen::NumericalDiff<MeanFunctor> num_diff(functor, 1e-6);

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<MeanFunctor>, double> lm(num_diff);
    int info = lm.minimize(x);

    std::cout << "current result: pitch & roll: " << x[0]/PI*180 << " " << x[1]/PI*180 << endl;

    pitch_cache_.push_back(x[0]);
    roll_cache_.push_back(x[1]);

    }
    void findMatchPoints(const cv::Mat img_left, const cv::Mat img_right, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2){
        cv::Mat grayimgLeft,grayimgRight;
        if(img_left.channels()==3&&img_right.channels()==3) {
            cv::cvtColor(img_left, grayimgLeft, CV_BGR2GRAY);
            cv::cvtColor(img_right, grayimgRight, CV_BGR2GRAY);
        }
        else{
            grayimgLeft=img_left;
            grayimgRight=img_right;
        }
        cv::Mat imgLeft,imgRight;
        float scale=0.5;
        cv::resize(grayimgLeft, imgLeft, cv::Size(grayimgLeft.cols * scale, grayimgLeft.rows * scale));
        cv::resize(grayimgRight, imgRight, cv::Size(grayimgRight.cols * scale, grayimgRight.rows * scale));
        cv::Ptr<cv::GFTTDetector>detector =cv::GFTTDetector::create();
        cv::Ptr<cv::DescriptorExtractor> extractor=cv::xfeatures2d::SIFT::create();
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

        vector<cv::KeyPoint>keypointsLeft,keypointsRight;
        cv::Mat descriptorsLeft, descriptorsRight;
        detector->detect(imgLeft,keypointsLeft);
        detector->detect(imgRight,keypointsRight);

        extractor->compute(imgLeft,keypointsLeft,descriptorsLeft);
        extractor->compute(imgRight,keypointsRight,descriptorsRight);

        vector<cv::DMatch>Matches,goodMatches;
        matcher->match(descriptorsLeft,descriptorsRight,Matches);

        int sz=Matches.size();
        double maxDist=0;double minDist=50;
        for(int i=0;i<sz;++i){
            double dist=Matches[i].distance;
            if(dist>maxDist) maxDist=dist;
            if(dist<minDist) minDist=dist;
        }
        for(auto iter=Matches.begin();iter!=Matches.end();++iter){
            if((*iter).distance<0.5*maxDist) 
                goodMatches.push_back(*iter);
        }
        for(auto iter=goodMatches.begin();iter!=goodMatches.end();++iter){
            int queryIdx=iter->queryIdx;
            int trainIdx=iter->trainIdx;
            cv::Point2f pt1 = keypointsLeft[queryIdx].pt/scale;
            cv::Point2f pt2 = keypointsRight[trainIdx].pt/scale;

            if(abs(pt1.y - pt2.y) <= UTILS_MATCH_MIN_DIST)
            {
                pts1.push_back(pt1);
                pts2.push_back(pt2);
            }
        }
        // 亚像素检测
        TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.01);
        cornerSubPix(grayimgLeft, pts1, cv::Size(5,5), cv::Size(-1, -1), criteria);
        cornerSubPix(grayimgRight, pts2, cv::Size(5,5), cv::Size(-1, -1), criteria);
        // Mat grayimg1, grayimg2;
        // if(3 == img_left.channels() )
        // {
        //     cvtColor(img_left, grayimg1, CV_BGR2GRAY);
        // } else {
        //     grayimg1 = img_left.clone();
        // }

        // if(3 == img_left.channels())
        // {
        //     cvtColor(img_right, grayimg2, CV_BGR2GRAY);
        // } else {
        //     grayimg2 = img_right.clone();
        // }

        // Mat img1, img2;
        // float scale = 0.5;

        // resize(grayimg1, img1, Size(grayimg1.cols * scale, grayimg1.rows * scale));
        // resize(grayimg2, img2, Size(grayimg2.cols * scale, grayimg2.rows * scale));

        // cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create();
        // Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
        // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

        // vector<KeyPoint> keypoints1, keypoints2;
        // Mat descriptors1, descriptors2;

        // detector->detect(img1, keypoints1);
        // detector->detect(img2, keypoints2);

        // extractor->compute(img1,keypoints1,descriptors1);
        // extractor->compute(img2,keypoints2,descriptors2);

        // vector<DMatch> matches, good_matches;
        // matcher->match(descriptors1, descriptors2, matches);
        // findGoodMatch(matches, good_matches);


        // for(int i = 0; i<good_matches.size();i++)
        // {
        //     int queryIdx = good_matches[i].queryIdx;
        //     int trainIdx = good_matches[i].trainIdx;

        //     Point2f pt1 = keypoints1[queryIdx].pt/scale;
        //     Point2f pt2 = keypoints2[trainIdx].pt/scale;

        //     if(abs(pt1.y - pt2.y) <= UTILS_MATCH_MIN_DIST)
        //     {
        //         pts1.push_back(pt1);
        //         pts2.push_back(pt2);
        //     }
        // }
        // // sub pixel
        // TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.01);

        // cornerSubPix(grayimg1, pts1, cv::Size(5,5), cv::Size(-1, -1), criteria);
        // cornerSubPix(grayimg2, pts2, cv::Size(5,5), cv::Size(-1, -1), criteria);

    }
    void findGoodMatch(std::vector<DMatch> matches, std::vector<DMatch> &good_matches)
    {
        int sz = matches.size();
        double max_dist = 0, min_dist = 50;

        for(int i=0; i < sz; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (int i = 0; i < sz; i++)
        {
            if(matches[i].distance < UTILS_FEATURE_MATCH_RATIO_THRESH * max_dist)
            {
                good_matches.push_back(matches[i]);
            }
        }
    }
    private:    



};
#endif
