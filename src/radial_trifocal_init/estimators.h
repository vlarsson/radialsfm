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

#include "util/types.h"

namespace colmap {
namespace init {

class RadialTrifocalTensorEstimator {
 public:
  // Triplet correspondences
  struct PointData {
    Eigen::Vector2d x1, x2, x3;
  };
  typedef PointData X_t;

  // Weight for the residuals
  typedef double Y_t;

  // Rotations of the cameras
  typedef std::vector<Eigen::Matrix3d> M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 6;

  // Estimate radial camera pose from 5 correspondences
  //
  // @param points2D   2D images points
  // @param points3D   3D world points
  //
  // @return           Camera pose as a 3x4 matrix.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points2D,
                                   const std::vector<Y_t>& weights);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D     2D image points as Nx2 matrix.
  // @param points3D     3D world points as Nx3 matrix.
  // @param proj_matrix  3x4 projection matrix.
  // @param residuals    Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& weights, const M_t& rotations,
                        std::vector<double>* residuals);
};

// Estimate radial trifocal tensor (intersecting principal axes)
// from 2D-2D-2D correspondences
bool EstimateRadialTrifocalTensor(
    const std::vector<Eigen::Vector2d>& points2D_1,
    const std::vector<Eigen::Vector2d>& points2D_2,
    const std::vector<Eigen::Vector2d>& points2D_3,
    std::vector<Eigen::Matrix3d>& rotations, size_t* num_inliers,
    std::vector<char>* inlier_mask);

}  // namespace init
}  // namespace colmap