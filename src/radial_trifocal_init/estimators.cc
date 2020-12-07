// Copyright (c) 2020, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Viktor Larsson

#include "radial_trifocal_init/estimators.h"
#include "base/pose.h"
#include "optim/loransac.h"
#include "radial_trifocal_init/tensor.h"
#include "util/misc.h"
namespace colmap {
namespace init {

std::vector<RadialTrifocalTensorEstimator::M_t>
RadialTrifocalTensorEstimator::Estimate(const std::vector<X_t>& points2D,
                                        const std::vector<Y_t>& weights) {
  Eigen::Matrix<double, 2, 6> x1, x2, x3;
  for (int i = 0; i < 6; ++i) {
    x1.col(i) = points2D[i].x1;
    x2.col(i) = points2D[i].x2;
    x3.col(i) = points2D[i].x3;
  }

  Eigen::Matrix<double, 8, 4> sols;
  int n_sols = SolveCalibRadialTrifocalTensor(x1, x2, x3, &sols);

  std::vector<M_t> output;

  for (int i = 0; i < n_sols; ++i) {
    Eigen::Matrix<double, 8, 1> tensor = sols.col(i);

    Eigen::Matrix<double, 2, 3> P1[2], P2[2], P3[2];
    int n_fact = FactorizeRadialTensor(tensor, P1, P2, P3);

    for (int j = 0; j < n_fact; ++j) {
      // Perform metric upgrade (guaranteed to exist, though might be complex!)
      Eigen::Matrix3d H;
      if (!MetricUpgradeRadial(P2[j], P3[j], &H)) {
        continue;  // no real upgrade
      }
      P2[j] = P2[j] * H;
      P3[j] = P3[j] * H;
      P2[j] = P2[j] / P2[j].row(0).norm();
      P3[j] = P3[j] / P3[j].row(0).norm();

      // There are two possible flips here
      //  1. Sign of each camera can be flipped independently (we keep first as
      //  eye(3) though)
      //  2. Flip along z axis in world space (i.e. diag(1,1,-1))
      // We go through each and only add the ones that have consistent
      // half-plane constraints

      std::vector<Eigen::Matrix3d> rotations(3);
      rotations[0].setIdentity();
      std::vector<double> residuals;

      for (int flip_z = 0; flip_z < 2; ++flip_z) {
        for (int flip2 = 0; flip2 < 2; ++flip2) {
          for (int flip3 = 0; flip3 < 2; ++flip3) {
            rotations[1].topRows<2>() = P2[j];
            rotations[2].topRows<2>() = P3[j];

            if (flip_z) {
              rotations[1].col(2) *= -1.0;
              rotations[2].col(2) *= -1.0;
            }
            if (flip2) {
              rotations[1] *= -1.0;
            }
            if (flip3) {
              rotations[2] *= -1.0;
            }

            // Complete to full rotations
            rotations[1].row(2) =
                rotations[1].row(0).cross(rotations[1].row(1));
            rotations[2].row(2) =
                rotations[2].row(0).cross(rotations[2].row(1));

            // To check the half-plane constraints we compute the residuals
            // which returns double_max if the constraints are violated
            Residuals(points2D, weights, rotations, &residuals);

            bool ok = true;
            for (double p : residuals) {
              ok &= p < 1e-3;  // these should be satisfied exactly (up to
                               // numerical instabilities)
            }
            if (ok) {
              output.push_back(rotations);
            }
          }
        }
      }
    }
  }

  return output;
}
void RadialTrifocalTensorEstimator::Residuals(const std::vector<X_t>& points2D,
                                              const std::vector<Y_t>& weights,
                                              const M_t& rotations,
                                              std::vector<double>* residuals) {
  CHECK_EQ(rotations.size(), 3);
  residuals->resize(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    // For residuals we triangulate the direction with two views and measure
    // error in the third
    const Eigen::Vector2d& x1 = points2D[i].x1;
    const Eigen::Vector2d& x2 = points2D[i].x2;
    const Eigen::Vector2d& x3 = points2D[i].x3;

    // Setup plane normals
    Eigen::Vector3d n1 =
        x1(1) * rotations[0].row(0) - x1(0) * rotations[0].row(1);
    Eigen::Vector3d n2 =
        x2(1) * rotations[1].row(0) - x2(0) * rotations[1].row(1);
    Eigen::Vector3d n3 =
        x3(1) * rotations[2].row(0) - x3(0) * rotations[2].row(1);
    n1.normalize();
    n2.normalize();
    n3.normalize();

    // Triangulate points (or rather directions)
    Eigen::Vector3d X12 = n1.cross(n2).normalized();
    Eigen::Vector3d X13 = n1.cross(n3).normalized();
    Eigen::Vector3d X23 = n2.cross(n3).normalized();

    // Correct the sign
    if (x1.dot(rotations[0].topRows<2>() * X12) < 0) {
      X12 *= -1.0;
    }
    if (x1.dot(rotations[0].topRows<2>() * X13) < 0) {
      X13 *= -1.0;
    }
    if (x2.dot(rotations[1].topRows<2>() * X23) < 0) {
      X23 *= -1.0;
    }

    // Check half-plane constraints in the other views
    if (x2.dot(rotations[1].topRows<2>() * X12) < 0 ||
        x3.dot(rotations[2].topRows<2>() * X12) < 0 ||
        x2.dot(rotations[1].topRows<2>() * X13) < 0 ||
        x3.dot(rotations[2].topRows<2>() * X13) < 0 ||
        x1.dot(rotations[0].topRows<2>() * X23) < 0 ||
        x3.dot(rotations[2].topRows<2>() * X23) < 0) {
      (*residuals)[i] = std::numeric_limits<double>::max();
      continue;
    }

    // Compute residuals in the view not used for triangulation
    Eigen::Vector2d p1 = (rotations[0].topRows<2>() * X23).normalized();
    Eigen::Vector2d p2 = (rotations[1].topRows<2>() * X13).normalized();
    Eigen::Vector2d p3 = (rotations[2].topRows<2>() * X12).normalized();

    double res1 = (x1 - p1.dot(x1) * p1).squaredNorm();
    double res2 = (x2 - p2.dot(x2) * p2).squaredNorm();
    double res3 = (x3 - p3.dot(x3) * p3).squaredNorm();

    (*residuals)[i] = weights[i] * (res1 + res2 + res3) / 3.0;
  }
}

// Estimate radial trifocal tensor (intersecting principal axes)
// from 2D-2D-2D correspondences
bool EstimateRadialTrifocalTensor(
    const std::vector<Eigen::Vector2d>& points2D_1,
    const std::vector<Eigen::Vector2d>& points2D_2,
    const std::vector<Eigen::Vector2d>& points2D_3,
    std::vector<Eigen::Matrix3d>& rotations, size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  std::vector<RadialTrifocalTensorEstimator::PointData> corrs;
  std::vector<double> weights;
  corrs.resize(points2D_1.size());
  weights.resize(points2D_1.size());

  for (size_t i = 0; i < points2D_1.size(); ++i) {
    corrs[i].x1 = points2D_1[i];
    corrs[i].x2 = points2D_2[i];
    corrs[i].x3 = points2D_3[i];
    weights[i] = 1.0;  // TODO: actually use the weights...
  }

  RANSACOptions options;
  options.max_error = 5.0;
  options.min_num_trials = 10000;
  options.max_num_trials = 100000;
  options.confidence = 0.9999;

  LORANSAC<RadialTrifocalTensorEstimator, RadialTrifocalTensorEstimator,
           MEstimatorSupportMeasurer>
      ransac(options);

  auto report = ransac.Estimate(corrs, weights);

  if (!report.success) {
    return false;
  }

  *num_inliers = report.support.num_inliers;
  *inlier_mask = report.inlier_mask;

  rotations = report.model;

  return true;
}




std::vector<MixedTrifocalTensorEstimator::M_t>
MixedTrifocalTensorEstimator::Estimate(const std::vector<X_t>& bearingVectors,
                                        const std::vector<Y_t>& points2D) {
  
  Eigen::Matrix<double, 3, 9> x_pinhole;
  Eigen::Matrix<double, 2, 9> x1, x2;
  for (int i = 0; i < 9; ++i) {
    x_pinhole.col(i) = bearingVectors[i];
    x1.col(i) = points2D[i].x1;
    x2.col(i) = points2D[i].x2;
  }

  Eigen::Matrix<double, 12, 48> sols;
  int n_sols = SolveCalibMixedTrifocalTensor(x_pinhole, x1, x2, &sols);

  std::vector<M_t> output;

  for (int i = 0; i < n_sols; ++i) {
    Eigen::Matrix<double, 12, 1> tensor = sols.col(i);

    Eigen::Matrix3x4d P1;
    Eigen::Matrix<double, 2, 4> P2, P3;
    FactorizeMixedTensor(tensor, &P1, &P2, &P3);

    Eigen::Matrix4d H;
    if(!MetricUpgradeMixed(P2, P3, &H)) {
        continue; // no real upgrade
    }

    P2 = P2 * H;
    P3 = P3 * H;
    P2 = P2 / P2.block<1,3>(0,0).norm();
    P3 = P3 / P3.block<1,3>(0,0).norm();

    std::vector<Eigen::Matrix3x4d> poses(3);
    poses[0].setIdentity();
    poses[0].col(3).setZero();    
    std::vector<double> residuals;

    // Complete to full rotations
    poses[1].topRows<2>() = P2;
    poses[2].topRows<2>() = P3;
    poses[1].block<1,3>(2,0) = poses[1].block<1,3>(0,0).cross(poses[1].block<1,3>(1,0));
    poses[2].block<1,3>(2,0) = poses[2].block<1,3>(0,0).cross(poses[2].block<1,3>(1,0));
    poses[1](2,3) = 0.0; // t3
    poses[2](2,3) = 0.0;

    // To check the half-plane constraints we compute the residuals
    // which returns double_max if the constraints are violated
    Residuals(bearingVectors, points2D, poses, &residuals);

    bool ok = true;
    for (double p : residuals) {
        ok &= p < 1e-3;  // these should be satisfied exactly (up to
                        // numerical instabilities)
    }
    if (ok) {
        output.push_back(poses);
    }        
  }

  return output;
}
void MixedTrifocalTensorEstimator::Residuals(const std::vector<X_t>& bearingVectors,
                                              const std::vector<Y_t>& points2D,
                                              const M_t& poses,
                                              std::vector<double>* residuals) {
  CHECK_EQ(poses.size(), 3);
  residuals->resize(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {

    // triangulate 3d point assuming zero error for pinhole camera

    const Eigen::Vector3d &X = bearingVectors[i];
    const Eigen::Vector2d &x1 = points2D[i].x1;
    const Eigen::Vector2d &x2 = points2D[i].x2;

    const Eigen::Vector2d RX1 = poses[1].block<2,3>(0,0) * X;
    const Eigen::Vector2d RX2 = poses[2].block<2,3>(0,0) * X;

    const double lambda1 = -(x1(1) * poses[1](0,3) - x1(0) * poses[1](1,3)) / (x1(1) * RX1(0) - x1(0) * RX1(1));
    const double lambda2 = -(x2(1) * poses[2](0,3) - x2(0) * poses[2](1,3)) / (x2(1) * RX2(0) - x2(0) * RX2(1));    
    const double lambda = (lambda1 + lambda2) / 2;
    const Eigen::Vector3d X_tri = lambda * X;

    Eigen::Vector2d p1 = (poses[1].topRows<2>() * X_tri.homogeneous()).normalized();
    Eigen::Vector2d p2 = (poses[2].topRows<2>() * X_tri.homogeneous()).normalized();

    if(lambda < 0 || x1.dot(p1) < 0 || x2.dot(p2) < 0) {
       (*residuals)[i] = std::numeric_limits<double>::max();
       continue;
    }
    
    const double res1 = (x1 - p1.dot(x1) * p1).squaredNorm();
    const double res2 = (x2 - p2.dot(x2) * p2).squaredNorm();

    (*residuals)[i] = (res1 + res2) / 2.0;
  }
}

// Estimate mixed trifocal tensor (central + 2 radial)
bool EstimateMixedTrifocalTensor(
    const std::vector<Eigen::Vector3d>& bearingVectors,
    const std::vector<Eigen::Vector2d>& points2D_1,
    const std::vector<Eigen::Vector2d>& points2D_2,
    std::vector<Eigen::Matrix3x4d>& poses, size_t* num_inliers,
    std::vector<char>* inlier_mask) {

  std::vector<MixedTrifocalTensorEstimator::PointData> points2D;  
  points2D.resize(points2D_1.size());

  for (size_t i = 0; i < points2D_1.size(); ++i) {
    points2D[i].x1 = points2D_1[i];
    points2D[i].x2 = points2D_2[i];    
  }

  RANSACOptions options;
  options.max_error = 5.0;
  options.min_num_trials = 10000;
  options.max_num_trials = 100000;
  options.confidence = 0.9999;

  LORANSAC<MixedTrifocalTensorEstimator, MixedTrifocalTensorEstimator,
           MEstimatorSupportMeasurer>
      ransac(options);

  auto report = ransac.Estimate(bearingVectors, points2D);

  if (!report.success) {
    return false;
  }

  *num_inliers = report.support.num_inliers;
  *inlier_mask = report.inlier_mask;

  poses = report.model;

  return true;
}

}  // namespace init
}  // namespace colmap