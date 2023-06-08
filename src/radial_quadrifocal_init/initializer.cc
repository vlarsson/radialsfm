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

#include "radial_quadrifocal_init/initializer.h"
#include "base/correspondence_graph.h"
#include "base/pose.h"
#include <radial_quadrifocal/rqt/quadrifocal_estimator.h>
#include <radial_quadrifocal/rqt/ransac_impl.h>


namespace colmap {
namespace rqt_init {

struct FeatureTrack {
  std::array<Eigen::Vector2d, 4> point2D;
};

std::vector<FeatureTrack> BuildQuadFeatureTracksForInitialization(
    const DatabaseCache& database_cache, const std::vector<image_t>& image_ids) {
  const auto& corr_graph = database_cache.CorrespondenceGraph();

  // Build reduced correspondence graph with only init images
  CorrespondenceGraph selected_corr_graph;

  for (const image_t image_id : image_ids) {
    selected_corr_graph.AddImage(image_id,
                                 database_cache.Image(image_id).NumPoints2D());
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < i; ++j) {
      selected_corr_graph.AddCorrespondences(
          image_ids[i], image_ids[j],
          corr_graph.FindCorrespondencesBetweenImages(image_ids[i],
                                                      image_ids[j]));
    }
  }

  std::unordered_map<image_t, int> image_id_idx_map;
  for (int i = 0; i < 4; ++i) {
    image_id_idx_map.emplace(image_ids.at(i), i);
  }

  // To find as many correspondences as possible, we go over all points in each
  // image
  std::set<std::array<int, 4>> track_set;
  for (const image_t image_id : image_ids) {
    const int num_points2D =
        static_cast<int>(database_cache.Image(image_id).NumPoints2D());
    for (int point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
      const auto& corrs =
          selected_corr_graph.FindCorrespondences(image_id, point2D_idx);

      std::array<int, 4> track;
      for (int i = 0; i < 4; ++i) {
        track[i] = kInvalidPoint2DIdx;
      }
      track[image_id_idx_map[image_id]] = point2D_idx;

      for (const auto& corr : corrs) {
        track[image_id_idx_map[corr.image_id]] = corr.point2D_idx;
      }

      bool quad_corr = true;
      for(int i = 0; i < 4; ++i) {
        if(track[i] == kInvalidPoint2DIdx) {
          quad_corr = false;
          break;
        }
      }
      if(quad_corr) {
        track_set.emplace(track);
      }
    }
  }

  // Go through each candidate track and filter out those which are not useful
  std::vector<FeatureTrack> tracks;
  for (const auto& track_ids : track_set) {
    FeatureTrack track;
    for (int i = 0; i < 4; ++i) {
      track.point2D[i] =
          database_cache.Image(image_ids[i]).Point2D(track_ids[i]).XY();
    }
    tracks.push_back(track);
  }
  return tracks;
}             


bool InitializeRadialQuadrifocal(const DatabaseCache& database_cache,
                                    const std::vector<image_t>& image_ids,
                                    std::vector<Eigen::Matrix3x4d> *poses) {
  CHECK_EQ(image_ids.size(), 4);

  std::vector<FeatureTrack> tracks =
      BuildQuadFeatureTracksForInitialization(database_cache, image_ids);

  // Remove principal point
  for (int i = 0; i < 4; ++i) {
    const Camera& camera =
        database_cache.Camera(database_cache.Image(image_ids[i]).CameraId());
    if (camera.ModelId() != Radial1DCameraModel::model_id) {
      std::cerr << "ERROR: Incorrect camera model! radial_quadrifocal_initializer "
                   "requires cameras to be 1D_RADIAL.\n";
      return false;
    }
    for (FeatureTrack& corr : tracks) {
      corr.point2D[i] = camera.ImageToWorld(corr.point2D[i]);
    }
  }

  std::cout << StringPrintf(
      "Found %d correspondences useful for initialization.\n", tracks.size());

  // Collect correspondences
  std::vector<Eigen::Vector2d> x1, x2, x3, x4;
  x1.reserve(tracks.size());
  x2.reserve(tracks.size());
  x3.reserve(tracks.size());
  x4.reserve(tracks.size());
  
  for (FeatureTrack& corr : tracks) {
    x1.push_back(corr.point2D[0]);
    x2.push_back(corr.point2D[1]);
    x3.push_back(corr.point2D[2]);
    x4.push_back(corr.point2D[3]);
  }


  rqt::StartSystem start_system;
  start_system.load_default(rqt::MinimalSolver::MINIMAL);
  rqt::TrackSettings track_settings;

  rqt::RansacOptions ransac_opt;
  ransac_opt.max_error = 5.0;
  ransac_opt.min_iterations = 100;
  ransac_opt.max_iterations = 10000;
  ransac_opt.solver = rqt::MinimalSolver::MINIMAL;
  rqt::QuadrifocalEstimator estimator(ransac_opt,x1,x2,x3,x4,start_system,track_settings);
  rqt::QuadrifocalEstimator::Reconstruction best_model;
  rqt::RansacStats stats = rqt::ransac(estimator, ransac_opt, &best_model);

  std::cout << StringPrintf(
      "RANSAC: %d/%d inliers (%3.2f %%).\n", stats.num_inliers, tracks.size(), stats.inlier_ratio*100);


  poses->resize(4);
  (*poses)[0].topRows<2>() = best_model.P1;
  (*poses)[1].topRows<2>() = best_model.P2;
  (*poses)[2].topRows<2>() = best_model.P3;
  (*poses)[3].topRows<2>() = best_model.P4;
  
  (*poses)[0].block<1,3>(2,0) = (*poses)[0].block<1,3>(0,0).cross((*poses)[0].block<1,3>(1,0));
  (*poses)[1].block<1,3>(2,0) = (*poses)[1].block<1,3>(0,0).cross((*poses)[1].block<1,3>(1,0));
  (*poses)[2].block<1,3>(2,0) = (*poses)[2].block<1,3>(0,0).cross((*poses)[2].block<1,3>(1,0));
  (*poses)[3].block<1,3>(2,0) = (*poses)[3].block<1,3>(0,0).cross((*poses)[3].block<1,3>(1,0));

  for(int i = 0; i < 4; ++i) {
    std::cout << "q" << i << " = " << RotationMatrixToQuaternion((*poses)[i].leftCols<3>()).transpose() << " " << (*poses)[i](0,3) << " " << (*poses)[i](1,3) << "\n";
  }

  // Check of the triangulated scene is completely planar (something probably went wrong)
  size_t cnt = 0;
  Eigen::Vector3d mu(0.0, 0.0, 0.0);
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for(size_t i = 0; i < best_model.X.size(); ++i) {
    if(best_model.inlier[i]) {
      cnt++;
      mu += best_model.X[i];
    }
  }
  mu /= static_cast<double>(cnt);
  for(size_t i = 0; i < best_model.X.size(); ++i) {
    if(best_model.inlier[i]) {
      Eigen::Vector3d c = best_model.X[i] - mu;
      cov += c * c.transpose();
    }
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov);
  Eigen::Vector3d s = svd.singularValues();

  bool is_planar = (s(2) / s(0)) < 0.01;

  if(is_planar) {
    std::cout << StringPrintf("Initialized scene is very close to planar (s3/s1 = %f)\n", s(2)/s(0));
  }

  return !is_planar;
}


}  // namespace init
}  // namespace colmap