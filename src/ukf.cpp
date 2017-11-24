#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const float UKF::CONST_DT = 1000000.0;

/**
 * Initializes Unscented Kalman filter
 */

UKF::~UKF() {}

UKF::UKF() {

  ///verify if the values were initalized
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1; //TODO: necessita ajustes

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1; //TODO: necessita ajustes

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.1;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 1;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.1;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.009;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  //state dimension
  n_x_ = 5;

  //augmented state dimension
  n_aug_ = 7;

  //z dimension
  n_z_ = 3;

  //define spreading parameter
  lambda_ = 3 - n_aug_; 

  //predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  //weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);   
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) { 
    weights_(i) = 0.5/(n_aug_ + lambda_);
  }

  //augmented sigma points
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);  
  Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  //last time
  time_us_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  H_laser_ = MatrixXd(2, 5);

  //measurement covariance matrix - laser
  R_laser_ << pow(std_laspx_,2), 0,
              0, pow(std_laspy_,2);

  H_laser_ << 1, 0, 
              0, 0,
              0, 1, 
              0, 0,
              0, 0;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(is_initialized_ == false){
    float px, py, rho, phi;
    cout<< "UKF:" << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_(0);
      py = meas_package.raw_measurements_(1);
    } 
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      rho = meas_package.raw_measurements_(0);
      phi = meas_package.raw_measurements_(1);
      px = rho * cos(phi);
      py = rho * sin(phi);
    }

    //update x, P and the last time
    this->x_ << px, py, 0, 0, 0;
    this->P_ = MatrixXd::Identity(n_x_, n_x_);
    this->time_us_ = meas_package.timestamp_;

    is_initialized_  = true;
  }
  else{
    //is initialized
    float dt = meas_package.timestamp_ - this->time_us_;
    float delta_t = dt / CONST_DT;
    this->time_us_ = meas_package.timestamp_;

    //predict values
    Prediction(delta_t);

    //update laser and radar measurements
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    } 
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  this->AugmentedSigmaPoints();
  this->PredictSigmaPoints(delta_t);
  this->PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  float px, py;

  if(this->use_laser_){
    px = meas_package.raw_measurements_(0);
    py = meas_package.raw_measurements_(1);

    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_pred = H_laser_ * x_;
    VectorXd y = z - z_pred;

    //estimate new values (same as EKF)
    MatrixXd PHt = P_ * H_laser_.transpose();  
    MatrixXd S = (H_laser_ * PHt) + R_laser_;
    MatrixXd K = PHt * S.inverse();

    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_laser_) * P_;
  }
  //else: ignore laser measurements
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  float rho, phi, px, py;
  float h1, h2, h3;

  if(this->use_radar_){
    rho = meas_package.raw_measurements_(0);
    phi = meas_package.raw_measurements_(1);
    px = rho * cos(phi);
    py = rho * sin(phi);

    //update mesurements
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_pred;
    MatrixXd S, Zsig;

    //predict radar
    PredictRadar(&z_pred, &S, &Zsig);

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = z - z_pred; //residual

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
  }
  //else: ignore radar measurements
}

/**
 * Create Augmented sigma points
*/
void UKF::AugmentedSigmaPoints() {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = pow(std_a_, 2);
  P_aug(6,6) = pow(std_yawdd_, 2);

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)         = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_)  = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

/**
 * Predict sigma points
*/
void UKF::PredictSigmaPoints(double delta_t) {
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++){
    //extract values for better readability
    double p_x      = Xsig_aug_(0,i);
    double p_y      = Xsig_aug_(1,i);
    double v        = Xsig_aug_(2,i);
    double yaw      = Xsig_aug_(3,i);
    double yawd     = Xsig_aug_(4,i);
    double nu_a     = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

/**
 * Predict mean and covariance
*/
void UKF::PredictMeanAndCovariance(){
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + (weights_(i) * Xsig_pred_.col(i));
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Predict radar
*/
void UKF::PredictRadar(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x  = Xsig_pred_(0,i);
    double p_y  = Xsig_pred_(1,i);
    double v    = Xsig_pred_(2,i);
    double yaw  = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R <<    pow(std_radr_,2), 0, 0,
          0, pow(std_radphi_,2), 0,
          0, 0, pow(std_radrd_,2);
  S = S + R;

  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}