// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/radial_absolute_pose"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/pose.h"
#include "estimators/radial_absolute_pose.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestRadialP5P) {
  SetPRNGSeed(0);

  for (int trials = 0; trials < 10; ++trials) {
    Eigen::Matrix3x4d gt_pose;
    gt_pose.block<3, 3>(0, 0) =
        Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    gt_pose.col(3).setRandom();

    std::vector<Eigen::Vector2d> points2D;
    std::vector<Eigen::Vector3d> points3D;

    for (int i = 0; i < 5; ++i) {
      Eigen::Vector2d x;
      x.setRandom();
      Eigen::Vector3d X = x.homogeneous();
      X *= RandomReal(0.1, 100.0);
      X = gt_pose.block<3, 3>(0, 0).transpose() * (X - gt_pose.col(3));

      points2D.push_back(x);
      points3D.push_back(X);
    }

    std::vector<Eigen::Matrix3x4d> poses =
        RadialP5PEstimator::Estimate(points2D, points3D);

    BOOST_CHECK(poses.size() > 0);

    double min_pose_diff = std::numeric_limits<double>::max();
    for (auto p : poses) {
      double diff = (gt_pose.block<2, 4>(0, 0) - p.block<2, 4>(0, 0)).norm();
      min_pose_diff = std::min(diff, min_pose_diff);

      std::vector<double> residuals;
      RadialP5PEstimator::Residuals(points2D, points3D, p, &residuals);

      for (double r : residuals) {
        BOOST_CHECK(r < 1e-6);
      }
    }

    BOOST_CHECK(min_pose_diff < 1e-2);
  }
}
