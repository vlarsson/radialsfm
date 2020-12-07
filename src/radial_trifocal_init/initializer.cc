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

#include "radial_trifocal_init/initializer.h"
#include "radial_trifocal_init/estimators.h"
#include "base/correspondence_graph.h"
#include "base/pose.h"

namespace colmap {
namespace init {


bool InitializeRadialReconstruction(const DatabaseCache &database_cache, const std::vector<image_t> &image_ids) {
  CHECK_EQ(image_ids.size(), 5);

  std::vector<FeatureTrack> tracks = BuildFeatureTracksForInitialization(database_cache, image_ids);
  
  // Remove principal point
  for(int i = 0; i < 5; ++i) {
      const Camera &camera = database_cache.Camera(database_cache.Image(image_ids[i]).CameraId());
      if(camera.ModelId() != Radial1DCameraModel::model_id) {
        std::cerr << "ERROR: Incorrect camera model! radial_trifocal_initializer requires cameras to be 1D_RADIAL.\n";
        return false;
      }
      for(FeatureTrack &corr : tracks) {
        if(corr.obs[i]) {
          corr.point2D[i] = camera.ImageToWorld(corr.point2D[i]);
        }
      }
  }

  std::cout << StringPrintf("Found %d correspondences useful for initialization.\n", tracks.size());

  std::vector<Eigen::Matrix3x4d> poses(5);

  if(!InitializeFirstTriplet(tracks, poses) {
    std::cout << "Unable to estimate radial trifocal tensor for the first three images. Maybe principal axes are not intersecting?\n";
    return false;
  }

  // We now have the poses of the first three cameras 
  // Next we triangulate a synthetic pinhole image
  // TODO


  return true;
}


std::vector<FeatureTrack> BuildFeatureTracksForInitialization(const DatabaseCache &database_cache, const std::vector<image_t> &image_ids) {
  const auto& corr_graph = database_cache.CorrespondenceGraph();

  // Build reduced correspondence graph with only init images
  CorrespondenceGraph selected_corr_graph;

  for (const image_t image_id : image_ids) {
    selected_corr_graph.AddImage(image_id, database_cache.Image(image_id).NumPoints2D());
  }

  for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < i; ++j) {
        selected_corr_graph.AddCorrespondences(image_ids[i], image_ids[j],
            corr_graph.FindCorrespondencesBetweenImages(image_ids[i], image_ids[j]));    
      }
  }

  std::unordered_map<image_t, int> image_id_idx_map;
  for (int i = 0; i < 4; ++i) {
    image_id_idx_map.emplace(image_ids.at(i), i);
  }

  // To find as many correspondences as possible, we go over all points in each image 
  std::set<std::array<int,5>> track_set;  
  for (const image_t image_id : image_ids) {
    const int num_points2D = static_cast<int>(database_cache.Image(image_id).NumPoints2D());
    for (int point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
      const auto& corrs = selected_corr_graph.FindCorrespondences(image_id, point2D_idx);

      std::array<int,5> track;
      for(int i = 0; i < 5; ++i) {
        track[i] = kInvalidPoint2DIdx;
      }      
      track[image_id_idx_map[image_id]] = point2D_idx;      

      for (const auto& corr : corrs) {
        track[image_id_idx_map[corr.image_id]] = corr.point2D_idx;
      }
      track_set.emplace(track);
    }
  }

  // Go through each candidate track and filter out those which are not useful
  std::vector<FeatureTrack> tracks;
  for(const auto &track_ids : track_set) {
    int obs_trifocal = 0;
    int obs_total = 0;

    FeatureTrack track;
    for(int i = 0; i < 5; ++i) {
        track.obs[i] = track_ids[i] != kInvalidPoint2DIdx;
        if(!track.obs[i]) {
            continue;
        }        
        track.point2D[i] = database_cache.Image(image_ids[i]).Point2D(track_ids[i]).XY();

        obs_total++;
        if(i < 3) {
            obs_trifocal++;
        }        
    }

    // We need either completely observed in the first three views (for trifocal est.)
    // or observed in 4+ views for general BA
    if(obs_trifocal == 3 || obs_total >= 4) {
        tracks.push_back(track);
    }    
  }

  return tracks;
}

    

bool InitializeFirstTriplet(const std::vector<FeatureTrack> &tracks, std::vector<Eigen::Matrix3x4d> &poses) {

  // Collect tracks that have observations in the first three images
  std::vector<Eigen::Vector2d> x1, x2, x3;
  for(size_t i = 0; i < tracks.size(); ++i) {
    if(tracks[i].obs[0] && tracks[i].obs[1] && tracks[i].obs[2]) {
      x1.push_back(tracks[i].point2D[0]);
      x2.push_back(tracks[i].point2D[1]);
      x3.push_back(tracks[i].point2D[2]);
    }
  }

  std::vector<Eigen::Matrix3d> rotations;
  std::vector<char> inlier_mask;
  size_t num_inliers;

  bool success = EstimateRadialTrifocalTensor(x1,x2,x3,rotations,&num_inliers,&inlier_mask);

  if(!success) {
    std::cout << "Estimation failed!\n";
    return false;
  }
  std::cout << StringPrintf("Estimated trifocal tensor with %d / %d inliers\n", num_inliers, x1.size());

  for(int i = 0; i < 3; ++i) {
    poses[i].leftCols<3>() = rotations[i];
    poses[i].col(3).setZero();
  }

  return true;
}


}
}