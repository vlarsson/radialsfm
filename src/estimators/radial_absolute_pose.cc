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

#include "estimators/radial_absolute_pose.h"
#include "estimators/utils.h"
#include "util/logging.h"

namespace colmap {
namespace {

// Stolen from PoseLib implementation
/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
inline double sign(const double z) { return z < 0 ? -1.0 : 1.0; }

int SolveQuadraticReal(double a, double b, double c, double roots[2]) {
  double b2m4ac = b * b - 4 * a * c;
  if (b2m4ac < 0) return 0;

  double sq = std::sqrt(b2m4ac);

  // Choose sign to avoid cancellations
  roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
  roots[1] = c / (a * roots[0]);

  return 2;
}
void SolveCubicRealSingleRoot(double c2, double c1, double c0, double& root) {
  double a = c1 - c2 * c2 / 3.0;
  double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
  double c = b * b / 4.0 + a * a * a / 27.0;
  if (c > 0) {
    c = std::sqrt(c);
    b *= -0.5;
    root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
  } else {
    c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
    root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
  }
}
/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
int SolveQuarticReal(double b, double c, double d, double e, double roots[4]) {
  // Find depressed quartic
  double p = c - 3.0 * b * b / 8.0;
  double q = b * b * b / 8.0 - 0.5 * b * c + d;
  double r =
      (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) /
      256.0;

  // Resolvent cubic is now
  // U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
  double bb = 2.0 * p;
  double cc = p * p - 4.0 * r;
  double dd = -q * q;

  // Solve resolvent cubic
  double u2;
  SolveCubicRealSingleRoot(bb, cc, dd, u2);

  if (u2 < 0) return 0;

  double u = sqrt(u2);

  double s = -u;
  double t = (p + u * u + q / u) / 2.0;
  double v = (p + u * u - q / u) / 2.0;

  int sols = 0;
  double disc = u * u - 4.0 * v;
  if (disc > 0) {
    roots[0] = (-u - sign(u) * std::sqrt(disc)) / 2.0;
    roots[1] = v / roots[0];
    sols += 2;
  }
  disc = s * s - 4.0 * t;
  if (disc > 0) {
    roots[sols] = (-s - sign(s) * std::sqrt(disc)) / 2.0;
    roots[sols + 1] = t / roots[sols];
    sols += 2;
  }

  for (int i = 0; i < sols; i++) {
    roots[i] = roots[i] - b / 4.0;

    // do one step of newton refinement
    double x = roots[i];
    double x2 = x * x;
    double x3 = x * x2;
    double dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) /
                (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
    roots[i] = x + dx;
  }
  return sols;
}
}  // namespace

std::vector<RadialP5PEstimator::M_t> RadialP5PEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  // We do some normalization on the input
  double focal0 = 0.0;
  for (int i = 0; i < 5; ++i) {
    focal0 += points2D[i].norm();
  }
  focal0 /= 5;

  std::vector<Eigen::Vector2d> scaled_points2D(5);
  for (int i = 0; i < 5; ++i) {
    scaled_points2D[i] = points2D[i] / focal0;
  }

  // Setup nullspace
  Eigen::Matrix<double, 8, 5> cc;
  for (int i = 0; i < 5; i++) {
    const double inv_s = 1.0 / points2D[i].norm();
    const double x = points2D[i](0) * inv_s;
    const double y = points2D[i](1) * inv_s;

    cc(0, i) = -y * points3D[i](0);
    cc(1, i) = -y * points3D[i](1);
    cc(2, i) = -y * points3D[i](2);
    cc(3, i) = -y;
    cc(4, i) = x * points3D[i](0);
    cc(5, i) = x * points3D[i](1);
    cc(6, i) = x * points3D[i](2);
    cc(7, i) = x;
  }

  Eigen::Matrix<double, 8, 8> Q = cc.householderQr().householderQ();
  Eigen::Matrix<double, 8, 3> N = Q.rightCols(3);

  // Compute coefficients for sylvester resultant
  double c11_1 = N(0, 1) * N(4, 1) + N(1, 1) * N(5, 1) + N(2, 1) * N(6, 1);
  double c12_1 = N(0, 1) * N(4, 2) + N(0, 2) * N(4, 1) + N(1, 1) * N(5, 2) +
                 N(1, 2) * N(5, 1) + N(2, 1) * N(6, 2) + N(2, 2) * N(6, 1);
  double c12_2 = N(0, 0) * N(4, 1) + N(0, 1) * N(4, 0) + N(1, 0) * N(5, 1) +
                 N(1, 1) * N(5, 0) + N(2, 0) * N(6, 1) + N(2, 1) * N(6, 0);
  double c13_1 = N(0, 2) * N(4, 2) + N(1, 2) * N(5, 2) + N(2, 2) * N(6, 2);
  double c13_2 = N(0, 0) * N(4, 2) + N(0, 2) * N(4, 0) + N(1, 0) * N(5, 2) +
                 N(1, 2) * N(5, 0) + N(2, 0) * N(6, 2) + N(2, 2) * N(6, 0);
  double c13_3 = N(0, 0) * N(4, 0) + N(1, 0) * N(5, 0) + N(2, 0) * N(6, 0);
  double c21_1 = N(0, 1) * N(0, 1) + N(1, 1) * N(1, 1) + N(2, 1) * N(2, 1) -
                 N(4, 1) * N(4, 1) - N(5, 1) * N(5, 1) - N(6, 1) * N(6, 1);
  double c22_1 = 2 * N(0, 1) * N(0, 2) + 2 * N(1, 1) * N(1, 2) +
                 2 * N(2, 1) * N(2, 2) - 2 * N(4, 1) * N(4, 2) -
                 2 * N(5, 1) * N(5, 2) - 2 * N(6, 1) * N(6, 2);
  double c22_2 = 2 * N(0, 0) * N(0, 1) + 2 * N(1, 0) * N(1, 1) +
                 2 * N(2, 0) * N(2, 1) - 2 * N(4, 0) * N(4, 1) -
                 2 * N(5, 0) * N(5, 1) - 2 * N(6, 0) * N(6, 1);
  double c23_1 = N(0, 2) * N(0, 2) + N(1, 2) * N(1, 2) + N(2, 2) * N(2, 2) -
                 N(4, 2) * N(4, 2) - N(5, 2) * N(5, 2) - N(6, 2) * N(6, 2);
  double c23_2 = 2 * N(0, 0) * N(0, 2) + 2 * N(1, 0) * N(1, 2) +
                 2 * N(2, 0) * N(2, 2) - 2 * N(4, 0) * N(4, 2) -
                 2 * N(5, 0) * N(5, 2) - 2 * N(6, 0) * N(6, 2);
  double c23_3 = N(0, 0) * N(0, 0) + N(1, 0) * N(1, 0) + N(2, 0) * N(2, 0) -
                 N(4, 0) * N(4, 0) - N(5, 0) * N(5, 0) - N(6, 0) * N(6, 0);

  double a4 = c11_1 * c11_1 * c23_3 * c23_3 - c11_1 * c12_2 * c22_2 * c23_3 -
              2 * c11_1 * c13_3 * c21_1 * c23_3 +
              c11_1 * c13_3 * c22_2 * c22_2 + c12_2 * c12_2 * c21_1 * c23_3 -
              c12_2 * c13_3 * c21_1 * c22_2 + c13_3 * c13_3 * c21_1 * c21_1;
  double a3 =
      c11_1 * c13_2 * c22_2 * c22_2 + 2 * c13_2 * c13_3 * c21_1 * c21_1 +
      c12_2 * c12_2 * c21_1 * c23_2 + 2 * c11_1 * c11_1 * c23_2 * c23_3 -
      c11_1 * c12_1 * c22_2 * c23_3 - c11_1 * c12_2 * c22_1 * c23_3 -
      c11_1 * c12_2 * c22_2 * c23_2 - 2 * c11_1 * c13_2 * c21_1 * c23_3 -
      2 * c11_1 * c13_3 * c21_1 * c23_2 + 2 * c11_1 * c13_3 * c22_1 * c22_2 +
      2 * c12_1 * c12_2 * c21_1 * c23_3 - c12_1 * c13_3 * c21_1 * c22_2 -
      c12_2 * c13_2 * c21_1 * c22_2 - c12_2 * c13_3 * c21_1 * c22_1;
  double a2 =
      c11_1 * c11_1 * c23_2 * c23_2 + c13_2 * c13_2 * c21_1 * c21_1 +
      c11_1 * c13_1 * c22_2 * c22_2 + c11_1 * c13_3 * c22_1 * c22_1 +
      2 * c13_1 * c13_3 * c21_1 * c21_1 + c12_2 * c12_2 * c21_1 * c23_1 +
      c12_1 * c12_1 * c21_1 * c23_3 + 2 * c11_1 * c11_1 * c23_1 * c23_3 -
      c11_1 * c12_1 * c22_1 * c23_3 - c11_1 * c12_1 * c22_2 * c23_2 -
      c11_1 * c12_2 * c22_1 * c23_2 - c11_1 * c12_2 * c22_2 * c23_1 -
      2 * c11_1 * c13_1 * c21_1 * c23_3 - 2 * c11_1 * c13_2 * c21_1 * c23_2 +
      2 * c11_1 * c13_2 * c22_1 * c22_2 - 2 * c11_1 * c13_3 * c21_1 * c23_1 +
      2 * c12_1 * c12_2 * c21_1 * c23_2 - c12_1 * c13_2 * c21_1 * c22_2 -
      c12_1 * c13_3 * c21_1 * c22_1 - c12_2 * c13_1 * c21_1 * c22_2 -
      c12_2 * c13_2 * c21_1 * c22_1;
  double a1 =
      c11_1 * c13_2 * c22_1 * c22_1 + 2 * c13_1 * c13_2 * c21_1 * c21_1 +
      c12_1 * c12_1 * c21_1 * c23_2 + 2 * c11_1 * c11_1 * c23_1 * c23_2 -
      c11_1 * c12_1 * c22_1 * c23_2 - c11_1 * c12_1 * c22_2 * c23_1 -
      c11_1 * c12_2 * c22_1 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_2 +
      2 * c11_1 * c13_1 * c22_1 * c22_2 - 2 * c11_1 * c13_2 * c21_1 * c23_1 +
      2 * c12_1 * c12_2 * c21_1 * c23_1 - c12_1 * c13_1 * c21_1 * c22_2 -
      c12_1 * c13_2 * c21_1 * c22_1 - c12_2 * c13_1 * c21_1 * c22_1;
  double a0 = c11_1 * c11_1 * c23_1 * c23_1 - c11_1 * c12_1 * c22_1 * c23_1 -
              2 * c11_1 * c13_1 * c21_1 * c23_1 +
              c11_1 * c13_1 * c22_1 * c22_1 + c12_1 * c12_1 * c21_1 * c23_1 -
              c12_1 * c13_1 * c21_1 * c22_1 + c13_1 * c13_1 * c21_1 * c21_1;

  a4 = 1.0 / a4;
  a3 *= a4;
  a2 *= a4;
  a1 *= a4;
  a0 *= a4;

  double roots[4];

  int n_roots = SolveQuarticReal(a3, a2, a1, a0, roots);

  std::vector<Eigen::Matrix3x4d> poses;

  for (int i = 0; i < n_roots; i++) {
    // We have two quadratic polynomials in y after substituting x
    double a = roots[i];
    double c1a = c11_1;
    double c1b = c12_1 + c12_2 * a;
    double c1c = c13_1 + c13_2 * a + c13_3 * a * a;

    double c2a = c21_1;
    double c2b = c22_1 + c22_2 * a;
    double c2c = c23_1 + c23_2 * a + c23_3 * a * a;

    // we solve the first one
    double bb[2];
    if (!SolveQuadraticReal(c1a, c1b, c1c, bb)) continue;

    // and check the residuals of the other
    double res1 = c2a * bb[0] * bb[0] + c2b * bb[0] + c2c;
    double res2;

    // For data where X(3,:) = 0 there is only a single solution
    // In this case the second solution will be NaN
    if (std::isnan(bb[1]))
      res2 = std::numeric_limits<double>::max();
    else
      res2 = c2a * bb[1] * bb[1] + c2b * bb[1] + c2c;

    double b = (std::abs(res1) > std::abs(res2)) ? bb[1] : bb[0];

    Eigen::Matrix<double, 8, 1> p = N.col(0) * a + N.col(1) * b + N.col(2);

    // This gives us the first two rows of the camera matrix
    Eigen::Matrix<double, 3, 4> pose;
    pose.row(0) << p(0), p(1), p(2), p(3);
    pose.row(1) << p(4), p(5), p(6), p(7);

    double scale = pose.block<1, 3>(0, 0).norm();
    pose.row(0) /= scale;
    pose.row(1) /= scale;
    pose.block<1, 3>(2, 0) =
        pose.block<1, 3>(0, 0).cross(pose.block<1, 3>(1, 0));

    // To avoid numerical troubles we project to rotation here. This adds some
    // slight runtime overhead but we might avoid some headaches later.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        pose.leftCols<3>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    pose.leftCols<3>() = svd.matrixU() * svd.matrixV().transpose();

    // Fix sign with first point correspondence
    if (points2D[0].dot(pose.block<2, 4>(0, 0) * points3D[0].homogeneous()) <
        0) {
      pose.block<2, 4>(0, 0) = -pose.block<2, 4>(0, 0);
    }

    // Check orientation constraints for all points
    bool ok = true;  // first point is already satisfied
    for (int k = 1; k < 5; ++k) {
      ok &= points2D[k].dot(pose.block<2, 4>(0, 0) *
                            points3D[k].homogeneous()) >= 0;
    }

    if (ok) {
      poses.push_back(pose);
    }
  }
  return poses;
}

void RadialP5PEstimator::Residuals(const std::vector<X_t>& points2D,
                                   const std::vector<Y_t>& points3D,
                                   const M_t& proj_matrix,
                                   std::vector<double>* residuals) {
  ComputeSquaredRadialReprojectionError(points2D, points3D, proj_matrix,
                                        residuals);
}

}  // namespace colmap
