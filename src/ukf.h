#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
  private:
    /**
     * Create Augmented sigma points
    */
    void AugmentedSigmaPoints();

    /**
     * Predict sigma points
    */
    void PredictSigmaPoints(double);

    /**
     * Predict mean and covariance
    */
    void PredictMeanAndCovariance();

    /**
     * Predict radar measurements
     * @param z_out Out parameter to return z_pred value
     * @param S_out Out parameter to return S matrix
     * @param Zsig_out Out parameter to return Zsig matrix
    */
    void PredictRadar(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out);

  public:

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_ = false;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///*augmented sigma points
    MatrixXd Xsig_aug_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_ ;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    ///* Z dimension
    int n_z_;

    ///* Sigma point spreading parameter
    double lambda_;

    ///* Constant to divide DT value 
    static const float CONST_DT;

    ///* Matrix R laser 
    MatrixXd R_laser_;

    ///* Matrix H laser
    MatrixXd H_laser_;


    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
