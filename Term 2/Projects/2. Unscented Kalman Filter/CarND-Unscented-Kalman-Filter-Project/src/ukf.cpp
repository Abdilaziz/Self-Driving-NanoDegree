#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;


  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6; // Tuning Param

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 6; // Tuning Param

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;


	is_initialized_ = false;

	time_us_ = 0.0;

	n_aug_ = 7; // 5 states and 2 noises to account for

	P_ << 1, 0, 0, 0, 0,
	  0, 1, 0, 0, 0,
	  0, 0, 1, 0, 0,
	  0, 0, 0, 1, 0,
	  0, 0, 0, 0, 1;

	// Xsig_pred_

	weights_ = VectorXd(2 * n_aug_ + 1);

	//create sigma point matrix  (7 x 15)
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	lambda_ = 3 - n_x_;

	// set weights
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i<2 * n_aug_ + 1; i++) {  //2n+1 weights
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

	NIS_laser = 0.0;
	NIS_radar = 0.0;


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (!is_initialized_) {

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

			double ro = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			double ro_dot = meas_package.raw_measurements_[2];
			double cphi = cos(phi);
			double sphi = sin(phi);

			// px, py, v, yaw, yaw_dot
			x_ << ro*cphi, ro*sphi, 0, phi, 0;

			time_us_ = meas_package.timestamp_;


		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

			double px = meas_package.raw_measurements_[0];
			double py = meas_package.raw_measurements_[1];

			x_ << px, py, 0, 0, 0;

			time_us_ = meas_package.timestamp_;

		}


		is_initialized_ = true;
		return;
	}



	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

	Prediction(dt);


	if ( (meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_    ) {

		UpdateLidar(meas_package);

	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {

		UpdateRadar(meas_package);

	}


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

	/*
		Create Augmented Sigma Points
	*/
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(n_x_) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(5, 5) = std_a_*std_a_;
	P_aug(6, 6) = std_yawdd_*std_yawdd_;


	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	/*
	Predict Sigma Points
	*/

	for (int i = 0; i < 2 * n_aug_ + 1; i++) 
	{

		double px = Xsig_aug(0, i);
		double py = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yaw_dot = Xsig_aug(4, i);
		double nu_accel = Xsig_aug(5, i);
		double nu_yaw_dd = Xsig_aug(6, i);

		// Different Update Function if yaw_dot is 0
		// old + update + noise

		double px_pred, py_pred, v_pred, yaw_pred, yaw_dot_pred;

		if (fabs(yaw_dot) > 0.001) {
			px_pred = px + (v / yaw_dot)*(sin(yaw + yaw_dot*delta_t) - sin(yaw)) + (0.5* (delta_t*delta_t)*cos(yaw)*nu_accel);
			py_pred = py + (v / yaw_dot)*(-cos(yaw - yaw_dot*delta_t) + cos(yaw)) + (0.5* (delta_t*delta_t)*sin(yaw)*nu_accel);
		}
		else {
			px_pred = px + v*cos(yaw)*delta_t + (0.5* (delta_t*delta_t)*cos(yaw)*nu_accel);
			py_pred = py + v*sin(yaw)*delta_t + (0.5* (delta_t*delta_t)*sin(yaw)*nu_accel);
		}

		v_pred = v + delta_t*nu_accel;
		yaw_pred = yaw + yaw_dot*delta_t  + (0.5*(delta_t*delta_t)*nu_yaw_dd);
		yaw_dot_pred = yaw_dot + delta_t*nu_yaw_dd;

		Xsig_pred_(0, i) = px_pred;
		Xsig_pred_(1, i) = py_pred;
		Xsig_pred_(2, i) = v_pred;
		Xsig_pred_(3, i) = yaw_pred;
		Xsig_pred_(4, i) = yaw_dot_pred;

	}


	/*
	Turn Sigma Points into Mean and Covariance Matrix
	*/

	// Calculate Mean of State
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	// Calculate Covariance Matrix of State
	for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
	VectorXd z_meas = meas_package.raw_measurements_;

	// Transform Predicted Sigma Points to Measurement Space
	int n_z_ = 2;
	MatrixXd Z_sigma = MatrixXd(n_z_, 2* n_aug_ +1);
	
	// For LDAR, only px and py are required
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		Z_sigma(0, i) = Xsig_pred_(0, i);
		Z_sigma(1, i) = Xsig_pred_(1, i);
	}

	// mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i)*Z_sigma.col(i);
	}

	//measurement covariance matrix S and Cross Correlation Matrix Tc
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) { 

		VectorXd z_diff = Z_sigma.col(i) - z_pred;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		S = S + weights_(i) * z_diff * z_diff.transpose();


		Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
	}

	MatrixXd R_laser = MatrixXd(n_z_, n_z_);
	R_laser << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	S = S + R_laser;

	/*
	update state with Measurement
	*/

	MatrixXd K = Tc*S.inverse();

	VectorXd z_error = z_meas - z_pred;

	x_ = x_ + K*z_error;

	P_ = P_ - K*S*K.transpose();

	NIS_laser = z_error.transpose()*S.inverse()*z_error;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
	VectorXd z_meas = meas_package.raw_measurements_;

	// Transform Predicted Sigma Points to Measurement Space
	int n_z_ = 3;
	MatrixXd Z_sigma = MatrixXd(n_z_, 2 * n_aug_ + 1);

	// For RADAR, we need rho, phi, and phi_dot
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double vx = cos(yaw)*v;
		double vy = sin(yaw)*v;

		// rho, phi, phi_dot
		double rho = sqrt(px*px + py*py);
		Z_sigma(0, i) = rho;
		Z_sigma(1, i) = atan2(py, px);
		Z_sigma(2, i) = (px*vx + py*vy) / rho;

	}

	// mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i)*Z_sigma.col(i);
	}

	//measurement covariance matrix S and Cross Correlation Matrix Tc
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Z_sigma.col(i) - z_pred;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
		Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
	}


	MatrixXd R_radar_ = MatrixXd(n_z_, n_z_);


	R_radar_ << std_radr_*std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;

	S = S + R_radar_;

	//Update state with new measurement

	MatrixXd K = Tc*S.inverse();

	VectorXd z_error = z_meas - z_pred;

	x_ = x_ + K*z_error;

	P_ = P_ - K*S*K.transpose();

	NIS_radar = z_error.transpose()*S.inverse()*z_error;

}
