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

#include <Eigen/Eigenvalues>

#include "base/polynomial.h"
#include "util/types.h"

#include "radial_trifocal_init/tensor.h"
#include "radial_trifocal_init/calib_mixed_coeffs.h"

namespace colmap {
namespace init {

void RadialTensorCoordinateChange(const Eigen::Matrix<double, 8, 1>& T,
                                  const Eigen::Matrix<double, 2, 2>& A1,
                                  const Eigen::Matrix<double, 2, 2>& A2,
                                  const Eigen::Matrix<double, 2, 2>& A3,
                                  Eigen::Matrix<double, 8, 1>* Tout) {
  (*Tout)(0) = A3(0) * (A2(0) * (A1(0) * T(0) + A1(1) * T(1)) +
                        A2(1) * (A1(0) * T(2) + A1(1) * T(3))) +
               A3(1) * (A2(0) * (A1(0) * T(4) + A1(1) * T(5)) +
                        A2(1) * (A1(0) * T(6) + A1(1) * T(7)));
  (*Tout)(1) = A3(0) * (A2(0) * (A1(2) * T(0) + A1(3) * T(1)) +
                        A2(1) * (A1(2) * T(2) + A1(3) * T(3))) +
               A3(1) * (A2(0) * (A1(2) * T(4) + A1(3) * T(5)) +
                        A2(1) * (A1(2) * T(6) + A1(3) * T(7)));
  (*Tout)(2) = A3(0) * (A2(2) * (A1(0) * T(0) + A1(1) * T(1)) +
                        A2(3) * (A1(0) * T(2) + A1(1) * T(3))) +
               A3(1) * (A2(2) * (A1(0) * T(4) + A1(1) * T(5)) +
                        A2(3) * (A1(0) * T(6) + A1(1) * T(7)));
  (*Tout)(3) = A3(0) * (A2(2) * (A1(2) * T(0) + A1(3) * T(1)) +
                        A2(3) * (A1(2) * T(2) + A1(3) * T(3))) +
               A3(1) * (A2(2) * (A1(2) * T(4) + A1(3) * T(5)) +
                        A2(3) * (A1(2) * T(6) + A1(3) * T(7)));
  (*Tout)(4) = A3(2) * (A2(0) * (A1(0) * T(0) + A1(1) * T(1)) +
                        A2(1) * (A1(0) * T(2) + A1(1) * T(3))) +
               A3(3) * (A2(0) * (A1(0) * T(4) + A1(1) * T(5)) +
                        A2(1) * (A1(0) * T(6) + A1(1) * T(7)));
  (*Tout)(5) = A3(2) * (A2(0) * (A1(2) * T(0) + A1(3) * T(1)) +
                        A2(1) * (A1(2) * T(2) + A1(3) * T(3))) +
               A3(3) * (A2(0) * (A1(2) * T(4) + A1(3) * T(5)) +
                        A2(1) * (A1(2) * T(6) + A1(3) * T(7)));
  (*Tout)(6) = A3(2) * (A2(2) * (A1(0) * T(0) + A1(1) * T(1)) +
                        A2(3) * (A1(0) * T(2) + A1(1) * T(3))) +
               A3(3) * (A2(2) * (A1(0) * T(4) + A1(1) * T(5)) +
                        A2(3) * (A1(0) * T(6) + A1(1) * T(7)));
  (*Tout)(7) = A3(2) * (A2(2) * (A1(2) * T(0) + A1(3) * T(1)) +
                        A2(3) * (A1(2) * T(2) + A1(3) * T(3))) +
               A3(3) * (A2(2) * (A1(2) * T(4) + A1(3) * T(5)) +
                        A2(3) * (A1(2) * T(6) + A1(3) * T(7)));
}
void MixedTensorCoordinateChange(const Eigen::Matrix<double, 12, 1>& T,
                                 const Eigen::Matrix<double, 3, 3>& A1,
                                 const Eigen::Matrix<double, 2, 2>& A2,
                                 const Eigen::Matrix<double, 2, 2>& A3,
                                 Eigen::Matrix<double, 12, 1>* Tout) {
  (*Tout)(0) = A3(0) * (A2(0) * (A1(0) * T(0) + A1(1) * T(1) + A1(2) * T(2)) +
                        A2(1) * (A1(0) * T(3) + A1(1) * T(4) + A1(2) * T(5))) +
               A3(1) * (A2(0) * (A1(0) * T(6) + A1(1) * T(7) + A1(2) * T(8)) +
                        A2(1) * (A1(0) * T(9) + A1(1) * T(10) + A1(2) * T(11)));
  (*Tout)(1) = A3(0) * (A2(0) * (A1(3) * T(0) + A1(4) * T(1) + A1(5) * T(2)) +
                        A2(1) * (A1(3) * T(3) + A1(4) * T(4) + A1(5) * T(5))) +
               A3(1) * (A2(0) * (A1(3) * T(6) + A1(4) * T(7) + A1(5) * T(8)) +
                        A2(1) * (A1(3) * T(9) + A1(4) * T(10) + A1(5) * T(11)));
  (*Tout)(2) = A3(0) * (A2(0) * (A1(6) * T(0) + A1(7) * T(1) + A1(8) * T(2)) +
                        A2(1) * (A1(6) * T(3) + A1(7) * T(4) + A1(8) * T(5))) +
               A3(1) * (A2(0) * (A1(6) * T(6) + A1(7) * T(7) + A1(8) * T(8)) +
                        A2(1) * (A1(6) * T(9) + A1(7) * T(10) + A1(8) * T(11)));
  (*Tout)(3) = A3(0) * (A2(2) * (A1(0) * T(0) + A1(1) * T(1) + A1(2) * T(2)) +
                        A2(3) * (A1(0) * T(3) + A1(1) * T(4) + A1(2) * T(5))) +
               A3(1) * (A2(2) * (A1(0) * T(6) + A1(1) * T(7) + A1(2) * T(8)) +
                        A2(3) * (A1(0) * T(9) + A1(1) * T(10) + A1(2) * T(11)));
  (*Tout)(4) = A3(0) * (A2(2) * (A1(3) * T(0) + A1(4) * T(1) + A1(5) * T(2)) +
                        A2(3) * (A1(3) * T(3) + A1(4) * T(4) + A1(5) * T(5))) +
               A3(1) * (A2(2) * (A1(3) * T(6) + A1(4) * T(7) + A1(5) * T(8)) +
                        A2(3) * (A1(3) * T(9) + A1(4) * T(10) + A1(5) * T(11)));
  (*Tout)(5) = A3(0) * (A2(2) * (A1(6) * T(0) + A1(7) * T(1) + A1(8) * T(2)) +
                        A2(3) * (A1(6) * T(3) + A1(7) * T(4) + A1(8) * T(5))) +
               A3(1) * (A2(2) * (A1(6) * T(6) + A1(7) * T(7) + A1(8) * T(8)) +
                        A2(3) * (A1(6) * T(9) + A1(7) * T(10) + A1(8) * T(11)));
  (*Tout)(6) = A3(2) * (A2(0) * (A1(0) * T(0) + A1(1) * T(1) + A1(2) * T(2)) +
                        A2(1) * (A1(0) * T(3) + A1(1) * T(4) + A1(2) * T(5))) +
               A3(3) * (A2(0) * (A1(0) * T(6) + A1(1) * T(7) + A1(2) * T(8)) +
                        A2(1) * (A1(0) * T(9) + A1(1) * T(10) + A1(2) * T(11)));
  (*Tout)(7) = A3(2) * (A2(0) * (A1(3) * T(0) + A1(4) * T(1) + A1(5) * T(2)) +
                        A2(1) * (A1(3) * T(3) + A1(4) * T(4) + A1(5) * T(5))) +
               A3(3) * (A2(0) * (A1(3) * T(6) + A1(4) * T(7) + A1(5) * T(8)) +
                        A2(1) * (A1(3) * T(9) + A1(4) * T(10) + A1(5) * T(11)));
  (*Tout)(8) = A3(2) * (A2(0) * (A1(6) * T(0) + A1(7) * T(1) + A1(8) * T(2)) +
                        A2(1) * (A1(6) * T(3) + A1(7) * T(4) + A1(8) * T(5))) +
               A3(3) * (A2(0) * (A1(6) * T(6) + A1(7) * T(7) + A1(8) * T(8)) +
                        A2(1) * (A1(6) * T(9) + A1(7) * T(10) + A1(8) * T(11)));
  (*Tout)(9) = A3(2) * (A2(2) * (A1(0) * T(0) + A1(1) * T(1) + A1(2) * T(2)) +
                        A2(3) * (A1(0) * T(3) + A1(1) * T(4) + A1(2) * T(5))) +
               A3(3) * (A2(2) * (A1(0) * T(6) + A1(1) * T(7) + A1(2) * T(8)) +
                        A2(3) * (A1(0) * T(9) + A1(1) * T(10) + A1(2) * T(11)));
  (*Tout)(10) =
      A3(2) * (A2(2) * (A1(3) * T(0) + A1(4) * T(1) + A1(5) * T(2)) +
               A2(3) * (A1(3) * T(3) + A1(4) * T(4) + A1(5) * T(5))) +
      A3(3) * (A2(2) * (A1(3) * T(6) + A1(4) * T(7) + A1(5) * T(8)) +
               A2(3) * (A1(3) * T(9) + A1(4) * T(10) + A1(5) * T(11)));
  (*Tout)(11) =
      A3(2) * (A2(2) * (A1(6) * T(0) + A1(7) * T(1) + A1(8) * T(2)) +
               A2(3) * (A1(6) * T(3) + A1(7) * T(4) + A1(8) * T(5))) +
      A3(3) * (A2(2) * (A1(6) * T(6) + A1(7) * T(7) + A1(8) * T(8)) +
               A2(3) * (A1(6) * T(9) + A1(7) * T(10) + A1(8) * T(11)));
}

bool MetricUpgradeRadial(const Eigen::Matrix<double, 2, 3>& P2,
                         const Eigen::Matrix<double, 2, 3>& P3,
                         Eigen::Matrix<double, 3, 3>* H) {
  Eigen::Matrix<double, 4, 3> A;
  Eigen::Matrix<double, 4, 1> b;
  A.row(0) << P2(4) * P2(4) - P2(5) * P2(5),
      2 * P2(0) * P2(4) - 2 * P2(1) * P2(5),
      2 * P2(2) * P2(4) - 2 * P2(3) * P2(5);
  b(0) = -P2(0) * P2(0) + P2(1) * P2(1) - P2(2) * P2(2) + P2(3) * P2(3);
  A.row(1) << P2(4) * P2(5), P2(0) * P2(5) + P2(1) * P2(4),
      P2(2) * P2(5) + P2(3) * P2(4);
  b(1) = -P2(0) * P2(1) - P2(2) * P2(3);
  A.row(2) << P3(4) * P3(4) - P3(5) * P3(5),
      2 * P3(0) * P3(4) - 2 * P3(1) * P3(5),
      2 * P3(2) * P3(4) - 2 * P3(3) * P3(5);
  b(2) = -P3(0) * P3(0) + P3(1) * P3(1) - P3(2) * P3(2) + P3(3) * P3(3);
  A.row(3) << P3(4) * P3(5), P3(0) * P3(5) + P3(1) * P3(4),
      P3(2) * P3(5) + P3(3) * P3(4);
  b(3) = -P3(0) * P3(1) - P3(2) * P3(3);

  Eigen::Matrix<double, 3, 1> vs = A.householderQr().solve(b);

  double v3_2 = vs(0) - vs(1) * vs(1) - vs(2) * vs(2);
  if (v3_2 <= 0) {
    return false;
  }

  double v3 = std::sqrt(v3_2);

  (*H) << 1, 0, 0, 0, 1, 0, vs(1), vs(2), v3;

  return true;
}

bool MetricUpgradeMixed(const Eigen::Matrix<double, 2, 4>& P2,
                        const Eigen::Matrix<double, 2, 4>& P3,
                        Eigen::Matrix<double, 4, 4>* H) {
  Eigen::Matrix<double, 4, 4> A;
  Eigen::Matrix<double, 4, 1> b;

  A.row(0) << P2(6) * P2(6) - P2(7) * P2(7),
      2 * P2(0) * P2(6) - 2 * P2(1) * P2(7),
      2 * P2(2) * P2(6) - 2 * P2(3) * P2(7),
      2 * P2(4) * P2(6) - 2 * P2(5) * P2(7);
  b(0) = -P2(0) * P2(0) + P2(1) * P2(1) - P2(2) * P2(2) + P2(3) * P2(3) -
         P2(4) * P2(4) + P2(5) * P2(5);
  A.row(1) << P2(6) * P2(7), P2(0) * P2(7) + P2(1) * P2(6),
      P2(2) * P2(7) + P2(3) * P2(6), P2(4) * P2(7) + P2(5) * P2(6);
  b(1) = -P2(0) * P2(1) - P2(2) * P2(3) - P2(4) * P2(5);

  A.row(2) << P3(6) * P3(6) - P3(7) * P3(7),
      2 * P3(0) * P3(6) - 2 * P3(1) * P3(7),
      2 * P3(2) * P3(6) - 2 * P3(3) * P3(7),
      2 * P3(4) * P3(6) - 2 * P3(5) * P3(7);
  b(2) = -P3(0) * P3(0) + P3(1) * P3(1) - P3(2) * P3(2) + P3(3) * P3(3) -
         P3(4) * P3(4) + P3(5) * P3(5);
  A.row(3) << P3(6) * P3(7), P3(0) * P3(7) + P3(1) * P3(6),
      P3(2) * P3(7) + P3(3) * P3(6), P3(4) * P3(7) + P3(5) * P3(6);
  b(3) = -P3(0) * P3(1) - P3(2) * P3(3) - P3(4) * P3(5);

  Eigen::Matrix<double, 4, 1> r2xyz = A.partialPivLu().solve(b);

  double tmp =
      (r2xyz.block<3, 1>(1, 0).squaredNorm() - r2xyz(0)) / r2xyz.squaredNorm();

  (*H) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, r2xyz(1), r2xyz(2), r2xyz(3), 1;

  return std::abs(tmp) < 1e-6;
}

int FactorizeRadialTensor(const Eigen::Matrix<double, 8, 1>& T,
                          Eigen::Matrix<double, 2, 3> P1[2],
                          Eigen::Matrix<double, 2, 3> P2[2],
                          Eigen::Matrix<double, 2, 3> P3[2]) {
  // This factorization method degenerates sometimes (e.g. pure rotation around
  // y axis), so we do a random projective change of variables in the images
  Eigen::Matrix<double, 8, 1> AT;
  Eigen::Matrix<double, 2, 2> A1;
  A1.setRandom();
  Eigen::Matrix<double, 2, 2> A2;
  A2.setRandom();
  Eigen::Matrix<double, 2, 2> A3;
  A3.setRandom();
  RadialTensorCoordinateChange(T, A1, A2, A3, &AT);

  double alpha = AT(2) * AT(7) - AT(3) * AT(6);
  double beta = AT(1) * AT(6) + AT(3) * AT(4) - AT(0) * AT(7) - AT(2) * AT(5);
  double gamma = AT(0) * AT(5) - AT(1) * AT(4);

  double aa1[2];
  int n_sols = SolveQuadraticReal(alpha, beta, gamma, aa1);

  Eigen::Matrix<double, 7, 6> G;

  for (int i = 0; i < n_sols; ++i) {
    double a1 = aa1[i];
    double s = std::sqrt(1 + a1 * a1);
    a1 /= s;
    double a2 = 1 / s;

    double rho = -(AT(1) * a2 - AT(3) * a1) / (AT(2) * a1 - AT(0) * a2);
    double b1 = rho * a1;
    double b2 = rho * a2;
    double c1 = -a2;
    double c2 = a1;

    G << 0, AT(7) * c2, -AT(0) * c1, 0, AT(0) * b1, -AT(7) * a2, 0, 0,
        -AT(1) * c1, AT(7) * c2, AT(1) * b1, -AT(7) * b2, 0, -AT(7) * c1,
        -AT(2) * c1, 0, AT(2) * b1, AT(7) * a1, 0, 0, -AT(3) * c1, -AT(7) * c1,
        AT(3) * b1, AT(7) * b1, -AT(7) * c2, 0, -AT(4) * c1, 0,
        AT(7) * a2 + AT(4) * b1, 0, 0, 0, -AT(5) * c1 - AT(7) * c2, 0,
        AT(7) * b2 + AT(5) * b1, 0, AT(7) * c1, 0, -AT(6) * c1, 0,
        -AT(7) * a1 + AT(6) * b1, 0;

    Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(G, Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 1> def = svd.matrixV().rightCols(1);

    P1[i] << 1, 0, 0, 0, 1, 0;
    P2[i] << a1, b1, c1, a2, b2, c2;
    P3[i] << def(0), def(2), def(4), def(1), def(3), def(5);
  }

  // Revert change of coordinates
  for (int i = 0; i < n_sols; ++i) {
    // P1[i] = A1 * P1[i];
    P2[i] = A2 * P2[i];
    P3[i] = A3 * P3[i];

    // Transform first camera back to [I2 0]
    P2[i].block<2, 2>(0, 0) *= A1.inverse();
    P3[i].block<2, 2>(0, 0) *= A1.inverse();
  }

  return n_sols;
}

void FactorizeMixedTensor(const Eigen::Matrix<double, 12, 1>& T,
                          Eigen::Matrix<double, 3, 4>* P1,
                          Eigen::Matrix<double, 2, 4>* P2,
                          Eigen::Matrix<double, 2, 4>* P3) {
  // This factorization method degenerates sometimes (e.g. pure rotation around
  // y axis), so we do a random projective change of variables in the images
  Eigen::Matrix<double, 12, 1> AT;
  Eigen::Matrix<double, 3, 3> A1;
  A1.setRandom();
  Eigen::Matrix<double, 2, 2> A2;
  A2.setRandom();
  Eigen::Matrix<double, 2, 2> A3;
  A3.setRandom();
  MixedTensorCoordinateChange(T, A1, A2, A3, &AT);

  AT /= -AT(11);
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> b;
  A << AT(6) * AT(10) * AT(10) + AT(8) * AT(9) * AT(10) * AT(10),
      AT(5) * AT(9) * AT(9) * AT(9) + AT(3) * AT(9) * AT(9),
      AT(8) * AT(10) * AT(10) * AT(10) + AT(7) * AT(10) * AT(10),
      AT(4) * AT(9) * AT(9) + AT(5) * AT(9) * AT(9) * AT(10);
  b << AT(0) * AT(9) * AT(10) +
           AT(9) * AT(9) * AT(10) * (AT(2) + AT(5) * AT(8)) +
           AT(3) * AT(8) * AT(9) * AT(10) + AT(5) * AT(6) * AT(9) * AT(10) +
           AT(5) * AT(8) * AT(9) * AT(9) * AT(10),
      AT(1) * AT(9) * AT(10) +
          AT(9) * AT(10) * AT(10) * (AT(2) + AT(5) * AT(8)) +
          AT(4) * AT(8) * AT(9) * AT(10) + AT(5) * AT(7) * AT(9) * AT(10) +
          AT(5) * AT(8) * AT(9) * AT(10) * AT(10);

  Eigen::Matrix<double, 2, 1> x = A.partialPivLu().solve(b);

  (*P1) << 1, 0, 0, 0, 0, 1, 0, 0, AT(9), AT(10), AT(9), AT(10);
  (*P2) << 0, 0, 1, 0, AT(8) + AT(6) / AT(9), (AT(8) * AT(10) + AT(7)) / AT(9),
      AT(8), x(1);
  (*P3) << 0, 0, 0, 1, (AT(5) * AT(9) + AT(3)) / AT(10), AT(5) + AT(4) / AT(10),
      x(0), AT(5);

  // Revert coordinate change
  (*P1) = A1 * (*P1);
  (*P2) = A2 * (*P2);
  (*P3) = A3 * (*P3);

  Eigen::Matrix<double, 4, 4> H;
  H.block<3, 4>(0, 0) = *P1;
  H.block<1, 4>(3, 0) << 0, 0, 0, 1.0;
  H = H.inverse().eval();

  (*P1) = (*P1) * H;
  (*P2) = (*P2) * H;
  (*P3) = (*P3) * H;
}

void EvaluateCalibTrifocalConstraintOnNullspace(
    const Eigen::Matrix<double, 8, 2>& N, Eigen::Array<double, 5, 1>* c) {
  (*c)(0) =
      N(1, 0) * (std::pow(N(6, 0), 3)) - N(0, 0) * (std::pow(N(7, 0), 3)) +
      N(2, 0) * (std::pow(N(5, 0), 3)) - N(3, 0) * (std::pow(N(4, 0), 3)) +
      (std::pow(N(0, 0), 3)) * N(7, 0) - (std::pow(N(1, 0), 3)) * N(6, 0) -
      (std::pow(N(2, 0), 3)) * N(5, 0) + (std::pow(N(3, 0), 3)) * N(4, 0) -
      (std::pow(N(0, 0), 2)) * N(1, 0) * N(6, 0) -
      (std::pow(N(0, 0), 2)) * N(2, 0) * N(5, 0) -
      (std::pow(N(0, 0), 2)) * N(3, 0) * N(4, 0) +
      N(0, 0) * (std::pow(N(1, 0), 2)) * N(7, 0) +
      (std::pow(N(1, 0), 2)) * N(2, 0) * N(5, 0) +
      (std::pow(N(1, 0), 2)) * N(3, 0) * N(4, 0) +
      N(0, 0) * (std::pow(N(2, 0), 2)) * N(7, 0) +
      N(1, 0) * (std::pow(N(2, 0), 2)) * N(6, 0) +
      (std::pow(N(2, 0), 2)) * N(3, 0) * N(4, 0) -
      N(0, 0) * (std::pow(N(3, 0), 2)) * N(7, 0) -
      N(1, 0) * (std::pow(N(3, 0), 2)) * N(6, 0) -
      N(2, 0) * (std::pow(N(3, 0), 2)) * N(5, 0) +
      N(0, 0) * (std::pow(N(4, 0), 2)) * N(7, 0) +
      N(1, 0) * (std::pow(N(4, 0), 2)) * N(6, 0) +
      N(2, 0) * (std::pow(N(4, 0), 2)) * N(5, 0) -
      N(0, 0) * (std::pow(N(5, 0), 2)) * N(7, 0) -
      N(1, 0) * (std::pow(N(5, 0), 2)) * N(6, 0) -
      N(3, 0) * N(4, 0) * (std::pow(N(5, 0), 2)) -
      N(0, 0) * (std::pow(N(6, 0), 2)) * N(7, 0) -
      N(2, 0) * N(5, 0) * (std::pow(N(6, 0), 2)) -
      N(3, 0) * N(4, 0) * (std::pow(N(6, 0), 2)) +
      N(1, 0) * N(6, 0) * (std::pow(N(7, 0), 2)) +
      N(2, 0) * N(5, 0) * (std::pow(N(7, 0), 2)) +
      N(3, 0) * N(4, 0) * (std::pow(N(7, 0), 2)) +
      2 * N(0, 0) * N(1, 0) * N(2, 0) * N(4, 0) -
      2 * N(0, 0) * N(1, 0) * N(3, 0) * N(5, 0) -
      2 * N(0, 0) * N(2, 0) * N(3, 0) * N(6, 0) +
      2 * N(1, 0) * N(2, 0) * N(3, 0) * N(7, 0) -
      2 * N(0, 0) * N(4, 0) * N(5, 0) * N(6, 0) +
      2 * N(1, 0) * N(4, 0) * N(5, 0) * N(7, 0) +
      2 * N(2, 0) * N(4, 0) * N(6, 0) * N(7, 0) -
      2 * N(3, 0) * N(5, 0) * N(6, 0) * N(7, 0);
  (*c)(1) =
      2 * N(4, 0) *
          (N(2, 0) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(0, 0) * N(1, 0) * N(2, 1)) -
      2 * N(5, 0) *
          (N(3, 0) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(0, 0) * N(1, 0) * N(3, 1)) -
      2 * N(6, 0) *
          (N(3, 0) * (N(0, 0) * N(2, 1) + N(2, 0) * N(0, 1)) +
           N(0, 0) * N(2, 0) * N(3, 1)) +
      2 * N(7, 0) *
          (N(3, 0) * (N(1, 0) * N(2, 1) + N(2, 0) * N(1, 1)) +
           N(1, 0) * N(2, 0) * N(3, 1)) -
      2 * N(6, 0) *
          (N(5, 0) * (N(0, 0) * N(4, 1) + N(4, 0) * N(0, 1)) +
           N(0, 0) * N(4, 0) * N(5, 1)) +
      2 * N(7, 0) *
          (N(5, 0) * (N(1, 0) * N(4, 1) + N(4, 0) * N(1, 1)) +
           N(1, 0) * N(4, 0) * N(5, 1)) +
      2 * N(7, 0) *
          (N(6, 0) * (N(2, 0) * N(4, 1) + N(4, 0) * N(2, 1)) +
           N(2, 0) * N(4, 0) * N(6, 1)) -
      2 * N(7, 0) *
          (N(6, 0) * (N(3, 0) * N(5, 1) + N(5, 0) * N(3, 1)) +
           N(3, 0) * N(5, 0) * N(6, 1)) +
      (std::pow(N(0, 0), 3)) * N(7, 1) - (std::pow(N(1, 0), 3)) * N(6, 1) -
      (std::pow(N(2, 0), 3)) * N(5, 1) + (std::pow(N(3, 0), 3)) * N(4, 1) -
      (std::pow(N(4, 0), 3)) * N(3, 1) + (std::pow(N(5, 0), 3)) * N(2, 1) +
      (std::pow(N(6, 0), 3)) * N(1, 1) - (std::pow(N(7, 0), 3)) * N(0, 1) -
      (std::pow(N(5, 0), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) -
      (std::pow(N(6, 0), 2)) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) -
      (std::pow(N(6, 0), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) +
      (std::pow(N(7, 0), 2)) * (N(1, 0) * N(6, 1) + N(6, 0) * N(1, 1)) +
      (std::pow(N(7, 0), 2)) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) +
      (std::pow(N(7, 0), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) -
      N(6, 0) *
          ((std::pow(N(0, 0), 2)) * N(1, 1) + 2 * N(0, 0) * N(1, 0) * N(0, 1)) -
      N(5, 0) *
          ((std::pow(N(0, 0), 2)) * N(2, 1) + 2 * N(0, 0) * N(2, 0) * N(0, 1)) -
      N(4, 0) *
          ((std::pow(N(0, 0), 2)) * N(3, 1) + 2 * N(0, 0) * N(3, 0) * N(0, 1)) +
      N(7, 0) *
          ((std::pow(N(1, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(1, 0) * N(1, 1)) +
      N(5, 0) *
          ((std::pow(N(1, 0), 2)) * N(2, 1) + 2 * N(1, 0) * N(2, 0) * N(1, 1)) +
      N(4, 0) *
          ((std::pow(N(1, 0), 2)) * N(3, 1) + 2 * N(1, 0) * N(3, 0) * N(1, 1)) +
      N(7, 0) *
          ((std::pow(N(2, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(2, 0) * N(2, 1)) +
      N(6, 0) *
          ((std::pow(N(2, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(2, 0) * N(2, 1)) +
      N(4, 0) *
          ((std::pow(N(2, 0), 2)) * N(3, 1) + 2 * N(2, 0) * N(3, 0) * N(2, 1)) -
      N(7, 0) *
          ((std::pow(N(3, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(3, 0) * N(3, 1)) -
      N(6, 0) *
          ((std::pow(N(3, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(3, 0) * N(3, 1)) -
      N(5, 0) *
          ((std::pow(N(3, 0), 2)) * N(2, 1) + 2 * N(2, 0) * N(3, 0) * N(3, 1)) +
      N(7, 0) *
          ((std::pow(N(4, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(4, 0) * N(4, 1)) +
      N(6, 0) *
          ((std::pow(N(4, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(4, 0) * N(4, 1)) +
      N(5, 0) *
          ((std::pow(N(4, 0), 2)) * N(2, 1) + 2 * N(2, 0) * N(4, 0) * N(4, 1)) -
      N(7, 0) *
          ((std::pow(N(5, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(5, 0) * N(5, 1)) -
      N(6, 0) *
          ((std::pow(N(5, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(5, 0) * N(5, 1)) -
      N(7, 0) *
          ((std::pow(N(6, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(6, 0) * N(6, 1)) -
      (std::pow(N(0, 0), 2)) * N(1, 0) * N(6, 1) -
      (std::pow(N(0, 0), 2)) * N(2, 0) * N(5, 1) -
      (std::pow(N(0, 0), 2)) * N(3, 0) * N(4, 1) +
      3 * (std::pow(N(0, 0), 2)) * N(7, 0) * N(0, 1) +
      N(0, 0) * (std::pow(N(1, 0), 2)) * N(7, 1) +
      (std::pow(N(1, 0), 2)) * N(2, 0) * N(5, 1) +
      (std::pow(N(1, 0), 2)) * N(3, 0) * N(4, 1) -
      3 * (std::pow(N(1, 0), 2)) * N(6, 0) * N(1, 1) +
      N(0, 0) * (std::pow(N(2, 0), 2)) * N(7, 1) +
      N(1, 0) * (std::pow(N(2, 0), 2)) * N(6, 1) +
      (std::pow(N(2, 0), 2)) * N(3, 0) * N(4, 1) -
      3 * (std::pow(N(2, 0), 2)) * N(5, 0) * N(2, 1) -
      N(0, 0) * (std::pow(N(3, 0), 2)) * N(7, 1) -
      N(1, 0) * (std::pow(N(3, 0), 2)) * N(6, 1) -
      N(2, 0) * (std::pow(N(3, 0), 2)) * N(5, 1) +
      3 * (std::pow(N(3, 0), 2)) * N(4, 0) * N(3, 1) +
      N(0, 0) * (std::pow(N(4, 0), 2)) * N(7, 1) +
      N(1, 0) * (std::pow(N(4, 0), 2)) * N(6, 1) +
      N(2, 0) * (std::pow(N(4, 0), 2)) * N(5, 1) -
      3 * N(3, 0) * (std::pow(N(4, 0), 2)) * N(4, 1) -
      N(0, 0) * (std::pow(N(5, 0), 2)) * N(7, 1) -
      N(1, 0) * (std::pow(N(5, 0), 2)) * N(6, 1) +
      3 * N(2, 0) * (std::pow(N(5, 0), 2)) * N(5, 1) -
      N(0, 0) * (std::pow(N(6, 0), 2)) * N(7, 1) +
      3 * N(1, 0) * (std::pow(N(6, 0), 2)) * N(6, 1) -
      3 * N(0, 0) * (std::pow(N(7, 0), 2)) * N(7, 1) +
      2 * N(0, 0) * N(1, 0) * N(2, 0) * N(4, 1) -
      2 * N(0, 0) * N(1, 0) * N(3, 0) * N(5, 1) -
      2 * N(0, 0) * N(2, 0) * N(3, 0) * N(6, 1) +
      2 * N(1, 0) * N(2, 0) * N(3, 0) * N(7, 1) -
      2 * N(0, 0) * N(4, 0) * N(5, 0) * N(6, 1) +
      2 * N(1, 0) * N(4, 0) * N(5, 0) * N(7, 1) -
      2 * N(3, 0) * N(4, 0) * N(5, 0) * N(5, 1) +
      2 * N(2, 0) * N(4, 0) * N(6, 0) * N(7, 1) -
      2 * N(2, 0) * N(5, 0) * N(6, 0) * N(6, 1) -
      2 * N(3, 0) * N(4, 0) * N(6, 0) * N(6, 1) +
      2 * N(1, 0) * N(6, 0) * N(7, 0) * N(7, 1) +
      2 * N(2, 0) * N(5, 0) * N(7, 0) * N(7, 1) +
      2 * N(3, 0) * N(4, 0) * N(7, 0) * N(7, 1) -
      2 * N(3, 0) * N(5, 0) * N(6, 0) * N(7, 1);
  (*c)(2) =
      2 * N(4, 1) *
          (N(2, 0) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(0, 0) * N(1, 0) * N(2, 1)) -
      2 * N(5, 1) *
          (N(3, 0) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(0, 0) * N(1, 0) * N(3, 1)) -
      2 * N(6, 1) *
          (N(3, 0) * (N(0, 0) * N(2, 1) + N(2, 0) * N(0, 1)) +
           N(0, 0) * N(2, 0) * N(3, 1)) +
      2 * N(4, 0) *
          (N(2, 1) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(2, 0) * N(0, 1) * N(1, 1)) +
      2 * N(7, 1) *
          (N(3, 0) * (N(1, 0) * N(2, 1) + N(2, 0) * N(1, 1)) +
           N(1, 0) * N(2, 0) * N(3, 1)) -
      2 * N(5, 0) *
          (N(3, 1) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(3, 0) * N(0, 1) * N(1, 1)) -
      2 * N(6, 0) *
          (N(3, 1) * (N(0, 0) * N(2, 1) + N(2, 0) * N(0, 1)) +
           N(3, 0) * N(0, 1) * N(2, 1)) -
      2 * N(6, 1) *
          (N(5, 0) * (N(0, 0) * N(4, 1) + N(4, 0) * N(0, 1)) +
           N(0, 0) * N(4, 0) * N(5, 1)) +
      2 * N(7, 0) *
          (N(3, 1) * (N(1, 0) * N(2, 1) + N(2, 0) * N(1, 1)) +
           N(3, 0) * N(1, 1) * N(2, 1)) +
      2 * N(7, 1) *
          (N(5, 0) * (N(1, 0) * N(4, 1) + N(4, 0) * N(1, 1)) +
           N(1, 0) * N(4, 0) * N(5, 1)) +
      2 * N(7, 1) *
          (N(6, 0) * (N(2, 0) * N(4, 1) + N(4, 0) * N(2, 1)) +
           N(2, 0) * N(4, 0) * N(6, 1)) -
      2 * N(6, 0) *
          (N(5, 1) * (N(0, 0) * N(4, 1) + N(4, 0) * N(0, 1)) +
           N(5, 0) * N(0, 1) * N(4, 1)) +
      2 * N(7, 0) *
          (N(5, 1) * (N(1, 0) * N(4, 1) + N(4, 0) * N(1, 1)) +
           N(5, 0) * N(1, 1) * N(4, 1)) -
      2 * N(7, 1) *
          (N(6, 0) * (N(3, 0) * N(5, 1) + N(5, 0) * N(3, 1)) +
           N(3, 0) * N(5, 0) * N(6, 1)) +
      2 * N(7, 0) *
          (N(6, 1) * (N(2, 0) * N(4, 1) + N(4, 0) * N(2, 1)) +
           N(6, 0) * N(2, 1) * N(4, 1)) -
      2 * N(7, 0) *
          (N(6, 1) * (N(3, 0) * N(5, 1) + N(5, 0) * N(3, 1)) +
           N(6, 0) * N(3, 1) * N(5, 1)) -
      N(6, 0) *
          (N(1, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(1, 1)) -
      N(6, 1) *
          ((std::pow(N(0, 0), 2)) * N(1, 1) + 2 * N(0, 0) * N(1, 0) * N(0, 1)) -
      N(5, 0) *
          (N(2, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(2, 1)) -
      N(5, 1) *
          ((std::pow(N(0, 0), 2)) * N(2, 1) + 2 * N(0, 0) * N(2, 0) * N(0, 1)) -
      N(4, 0) *
          (N(3, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(3, 1)) +
      N(7, 0) *
          (N(0, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(0, 1) * N(1, 1)) -
      N(4, 1) *
          ((std::pow(N(0, 0), 2)) * N(3, 1) + 2 * N(0, 0) * N(3, 0) * N(0, 1)) +
      N(7, 1) *
          ((std::pow(N(1, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(1, 0) * N(1, 1)) +
      N(5, 0) *
          (N(2, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(1, 1) * N(2, 1)) +
      N(5, 1) *
          ((std::pow(N(1, 0), 2)) * N(2, 1) + 2 * N(1, 0) * N(2, 0) * N(1, 1)) +
      N(4, 0) *
          (N(3, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(1, 1) * N(3, 1)) +
      N(7, 0) *
          (N(0, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(0, 1) * N(2, 1)) +
      N(4, 1) *
          ((std::pow(N(1, 0), 2)) * N(3, 1) + 2 * N(1, 0) * N(3, 0) * N(1, 1)) +
      N(7, 1) *
          ((std::pow(N(2, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(2, 0) * N(2, 1)) +
      N(6, 0) *
          (N(1, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(1, 1) * N(2, 1)) +
      N(6, 1) *
          ((std::pow(N(2, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(2, 0) * N(2, 1)) +
      N(4, 0) *
          (N(3, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(2, 1) * N(3, 1)) -
      N(7, 0) *
          (N(0, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(0, 1) * N(3, 1)) +
      N(4, 1) *
          ((std::pow(N(2, 0), 2)) * N(3, 1) + 2 * N(2, 0) * N(3, 0) * N(2, 1)) -
      N(7, 1) *
          ((std::pow(N(3, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(3, 0) * N(3, 1)) -
      N(6, 0) *
          (N(1, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(1, 1) * N(3, 1)) -
      N(6, 1) *
          ((std::pow(N(3, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(3, 0) * N(3, 1)) -
      N(5, 0) *
          (N(2, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(2, 1) * N(3, 1)) -
      N(5, 1) *
          ((std::pow(N(3, 0), 2)) * N(2, 1) + 2 * N(2, 0) * N(3, 0) * N(3, 1)) +
      N(7, 0) *
          (N(0, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(0, 1) * N(4, 1)) +
      N(7, 1) *
          ((std::pow(N(4, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(4, 0) * N(4, 1)) +
      N(6, 0) *
          (N(1, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(1, 1) * N(4, 1)) +
      N(6, 1) *
          ((std::pow(N(4, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(4, 0) * N(4, 1)) +
      N(5, 0) *
          (N(2, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(2, 1) * N(4, 1)) +
      N(5, 1) *
          ((std::pow(N(4, 0), 2)) * N(2, 1) + 2 * N(2, 0) * N(4, 0) * N(4, 1)) -
      N(7, 0) *
          (N(0, 0) * (std::pow(N(5, 1), 2)) + 2 * N(5, 0) * N(0, 1) * N(5, 1)) -
      N(7, 1) *
          ((std::pow(N(5, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(5, 0) * N(5, 1)) -
      N(6, 0) *
          (N(1, 0) * (std::pow(N(5, 1), 2)) + 2 * N(5, 0) * N(1, 1) * N(5, 1)) -
      N(6, 1) *
          ((std::pow(N(5, 0), 2)) * N(1, 1) + 2 * N(1, 0) * N(5, 0) * N(5, 1)) -
      N(7, 0) *
          (N(0, 0) * (std::pow(N(6, 1), 2)) + 2 * N(6, 0) * N(0, 1) * N(6, 1)) -
      N(7, 1) *
          ((std::pow(N(6, 0), 2)) * N(0, 1) + 2 * N(0, 0) * N(6, 0) * N(6, 1)) -
      2 * N(5, 0) * N(5, 1) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) -
      2 * N(6, 0) * N(6, 1) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) -
      2 * N(6, 0) * N(6, 1) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) +
      2 * N(7, 0) * N(7, 1) * (N(1, 0) * N(6, 1) + N(6, 0) * N(1, 1)) +
      2 * N(7, 0) * N(7, 1) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) +
      2 * N(7, 0) * N(7, 1) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) +
      3 * N(0, 0) * N(7, 0) * (std::pow(N(0, 1), 2)) -
      3 * N(1, 0) * N(6, 0) * (std::pow(N(1, 1), 2)) -
      3 * N(2, 0) * N(5, 0) * (std::pow(N(2, 1), 2)) +
      3 * N(3, 0) * N(4, 0) * (std::pow(N(3, 1), 2)) -
      3 * N(3, 0) * N(4, 0) * (std::pow(N(4, 1), 2)) +
      3 * N(2, 0) * N(5, 0) * (std::pow(N(5, 1), 2)) -
      N(3, 0) * N(4, 0) * (std::pow(N(5, 1), 2)) +
      3 * N(1, 0) * N(6, 0) * (std::pow(N(6, 1), 2)) -
      N(2, 0) * N(5, 0) * (std::pow(N(6, 1), 2)) -
      N(3, 0) * N(4, 0) * (std::pow(N(6, 1), 2)) -
      3 * N(0, 0) * N(7, 0) * (std::pow(N(7, 1), 2)) +
      N(1, 0) * N(6, 0) * (std::pow(N(7, 1), 2)) +
      N(2, 0) * N(5, 0) * (std::pow(N(7, 1), 2)) +
      N(3, 0) * N(4, 0) * (std::pow(N(7, 1), 2)) +
      3 * (std::pow(N(0, 0), 2)) * N(0, 1) * N(7, 1) -
      3 * (std::pow(N(1, 0), 2)) * N(1, 1) * N(6, 1) -
      3 * (std::pow(N(2, 0), 2)) * N(2, 1) * N(5, 1) +
      3 * (std::pow(N(3, 0), 2)) * N(3, 1) * N(4, 1) -
      3 * (std::pow(N(4, 0), 2)) * N(3, 1) * N(4, 1) +
      3 * (std::pow(N(5, 0), 2)) * N(2, 1) * N(5, 1) -
      (std::pow(N(5, 0), 2)) * N(3, 1) * N(4, 1) +
      3 * (std::pow(N(6, 0), 2)) * N(1, 1) * N(6, 1) -
      (std::pow(N(6, 0), 2)) * N(2, 1) * N(5, 1) -
      (std::pow(N(6, 0), 2)) * N(3, 1) * N(4, 1) -
      3 * (std::pow(N(7, 0), 2)) * N(0, 1) * N(7, 1) +
      (std::pow(N(7, 0), 2)) * N(1, 1) * N(6, 1) +
      (std::pow(N(7, 0), 2)) * N(2, 1) * N(5, 1) +
      (std::pow(N(7, 0), 2)) * N(3, 1) * N(4, 1);
  (*c)(3) =
      2 * N(4, 1) *
          (N(2, 1) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(2, 0) * N(0, 1) * N(1, 1)) -
      2 * N(5, 1) *
          (N(3, 1) * (N(0, 0) * N(1, 1) + N(1, 0) * N(0, 1)) +
           N(3, 0) * N(0, 1) * N(1, 1)) -
      2 * N(6, 1) *
          (N(3, 1) * (N(0, 0) * N(2, 1) + N(2, 0) * N(0, 1)) +
           N(3, 0) * N(0, 1) * N(2, 1)) +
      2 * N(7, 1) *
          (N(3, 1) * (N(1, 0) * N(2, 1) + N(2, 0) * N(1, 1)) +
           N(3, 0) * N(1, 1) * N(2, 1)) -
      2 * N(6, 1) *
          (N(5, 1) * (N(0, 0) * N(4, 1) + N(4, 0) * N(0, 1)) +
           N(5, 0) * N(0, 1) * N(4, 1)) +
      2 * N(7, 1) *
          (N(5, 1) * (N(1, 0) * N(4, 1) + N(4, 0) * N(1, 1)) +
           N(5, 0) * N(1, 1) * N(4, 1)) +
      2 * N(7, 1) *
          (N(6, 1) * (N(2, 0) * N(4, 1) + N(4, 0) * N(2, 1)) +
           N(6, 0) * N(2, 1) * N(4, 1)) -
      2 * N(7, 1) *
          (N(6, 1) * (N(3, 0) * N(5, 1) + N(5, 0) * N(3, 1)) +
           N(6, 0) * N(3, 1) * N(5, 1)) -
      N(0, 0) * (std::pow(N(7, 1), 3)) + N(1, 0) * (std::pow(N(6, 1), 3)) +
      N(2, 0) * (std::pow(N(5, 1), 3)) - N(3, 0) * (std::pow(N(4, 1), 3)) +
      N(4, 0) * (std::pow(N(3, 1), 3)) - N(5, 0) * (std::pow(N(2, 1), 3)) -
      N(6, 0) * (std::pow(N(1, 1), 3)) + N(7, 0) * (std::pow(N(0, 1), 3)) -
      (std::pow(N(5, 1), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) -
      (std::pow(N(6, 1), 2)) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) -
      (std::pow(N(6, 1), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) +
      (std::pow(N(7, 1), 2)) * (N(1, 0) * N(6, 1) + N(6, 0) * N(1, 1)) +
      (std::pow(N(7, 1), 2)) * (N(2, 0) * N(5, 1) + N(5, 0) * N(2, 1)) +
      (std::pow(N(7, 1), 2)) * (N(3, 0) * N(4, 1) + N(4, 0) * N(3, 1)) -
      N(6, 1) *
          (N(1, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(1, 1)) -
      N(5, 1) *
          (N(2, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(2, 1)) -
      N(4, 1) *
          (N(3, 0) * (std::pow(N(0, 1), 2)) + 2 * N(0, 0) * N(0, 1) * N(3, 1)) +
      N(7, 1) *
          (N(0, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(0, 1) * N(1, 1)) +
      N(5, 1) *
          (N(2, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(1, 1) * N(2, 1)) +
      N(4, 1) *
          (N(3, 0) * (std::pow(N(1, 1), 2)) + 2 * N(1, 0) * N(1, 1) * N(3, 1)) +
      N(7, 1) *
          (N(0, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(0, 1) * N(2, 1)) +
      N(6, 1) *
          (N(1, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(1, 1) * N(2, 1)) +
      N(4, 1) *
          (N(3, 0) * (std::pow(N(2, 1), 2)) + 2 * N(2, 0) * N(2, 1) * N(3, 1)) -
      N(7, 1) *
          (N(0, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(0, 1) * N(3, 1)) -
      N(6, 1) *
          (N(1, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(1, 1) * N(3, 1)) -
      N(5, 1) *
          (N(2, 0) * (std::pow(N(3, 1), 2)) + 2 * N(3, 0) * N(2, 1) * N(3, 1)) +
      N(7, 1) *
          (N(0, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(0, 1) * N(4, 1)) +
      N(6, 1) *
          (N(1, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(1, 1) * N(4, 1)) +
      N(5, 1) *
          (N(2, 0) * (std::pow(N(4, 1), 2)) + 2 * N(4, 0) * N(2, 1) * N(4, 1)) -
      N(7, 1) *
          (N(0, 0) * (std::pow(N(5, 1), 2)) + 2 * N(5, 0) * N(0, 1) * N(5, 1)) -
      N(6, 1) *
          (N(1, 0) * (std::pow(N(5, 1), 2)) + 2 * N(5, 0) * N(1, 1) * N(5, 1)) -
      N(7, 1) *
          (N(0, 0) * (std::pow(N(6, 1), 2)) + 2 * N(6, 0) * N(0, 1) * N(6, 1)) +
      3 * N(0, 0) * (std::pow(N(0, 1), 2)) * N(7, 1) -
      N(4, 0) * (std::pow(N(0, 1), 2)) * N(3, 1) -
      N(5, 0) * (std::pow(N(0, 1), 2)) * N(2, 1) -
      N(6, 0) * (std::pow(N(0, 1), 2)) * N(1, 1) -
      3 * N(1, 0) * (std::pow(N(1, 1), 2)) * N(6, 1) +
      N(4, 0) * (std::pow(N(1, 1), 2)) * N(3, 1) +
      N(5, 0) * (std::pow(N(1, 1), 2)) * N(2, 1) +
      N(7, 0) * N(0, 1) * (std::pow(N(1, 1), 2)) -
      3 * N(2, 0) * (std::pow(N(2, 1), 2)) * N(5, 1) +
      N(4, 0) * (std::pow(N(2, 1), 2)) * N(3, 1) +
      N(6, 0) * N(1, 1) * (std::pow(N(2, 1), 2)) +
      N(7, 0) * N(0, 1) * (std::pow(N(2, 1), 2)) +
      3 * N(3, 0) * (std::pow(N(3, 1), 2)) * N(4, 1) -
      N(5, 0) * N(2, 1) * (std::pow(N(3, 1), 2)) -
      N(6, 0) * N(1, 1) * (std::pow(N(3, 1), 2)) -
      N(7, 0) * N(0, 1) * (std::pow(N(3, 1), 2)) -
      3 * N(4, 0) * N(3, 1) * (std::pow(N(4, 1), 2)) +
      N(5, 0) * N(2, 1) * (std::pow(N(4, 1), 2)) +
      N(6, 0) * N(1, 1) * (std::pow(N(4, 1), 2)) +
      N(7, 0) * N(0, 1) * (std::pow(N(4, 1), 2)) +
      3 * N(5, 0) * N(2, 1) * (std::pow(N(5, 1), 2)) -
      N(6, 0) * N(1, 1) * (std::pow(N(5, 1), 2)) -
      N(7, 0) * N(0, 1) * (std::pow(N(5, 1), 2)) +
      3 * N(6, 0) * N(1, 1) * (std::pow(N(6, 1), 2)) -
      N(7, 0) * N(0, 1) * (std::pow(N(6, 1), 2)) -
      3 * N(7, 0) * N(0, 1) * (std::pow(N(7, 1), 2)) +
      2 * N(4, 0) * N(0, 1) * N(1, 1) * N(2, 1) -
      2 * N(5, 0) * N(0, 1) * N(1, 1) * N(3, 1) -
      2 * N(6, 0) * N(0, 1) * N(2, 1) * N(3, 1) +
      2 * N(7, 0) * N(1, 1) * N(2, 1) * N(3, 1) -
      2 * N(6, 0) * N(0, 1) * N(4, 1) * N(5, 1) -
      2 * N(5, 0) * N(3, 1) * N(4, 1) * N(5, 1) +
      2 * N(7, 0) * N(1, 1) * N(4, 1) * N(5, 1) -
      2 * N(6, 0) * N(2, 1) * N(5, 1) * N(6, 1) -
      2 * N(6, 0) * N(3, 1) * N(4, 1) * N(6, 1) +
      2 * N(7, 0) * N(2, 1) * N(4, 1) * N(6, 1) +
      2 * N(7, 0) * N(1, 1) * N(6, 1) * N(7, 1) +
      2 * N(7, 0) * N(2, 1) * N(5, 1) * N(7, 1) +
      2 * N(7, 0) * N(3, 1) * N(4, 1) * N(7, 1) -
      2 * N(7, 0) * N(3, 1) * N(5, 1) * N(6, 1);
  (*c)(4) =
      N(1, 1) * (std::pow(N(6, 1), 3)) - N(0, 1) * (std::pow(N(7, 1), 3)) +
      N(2, 1) * (std::pow(N(5, 1), 3)) - N(3, 1) * (std::pow(N(4, 1), 3)) +
      (std::pow(N(0, 1), 3)) * N(7, 1) - (std::pow(N(1, 1), 3)) * N(6, 1) -
      (std::pow(N(2, 1), 3)) * N(5, 1) + (std::pow(N(3, 1), 3)) * N(4, 1) -
      (std::pow(N(0, 1), 2)) * N(1, 1) * N(6, 1) -
      (std::pow(N(0, 1), 2)) * N(2, 1) * N(5, 1) -
      (std::pow(N(0, 1), 2)) * N(3, 1) * N(4, 1) +
      N(0, 1) * (std::pow(N(1, 1), 2)) * N(7, 1) +
      (std::pow(N(1, 1), 2)) * N(2, 1) * N(5, 1) +
      (std::pow(N(1, 1), 2)) * N(3, 1) * N(4, 1) +
      N(0, 1) * (std::pow(N(2, 1), 2)) * N(7, 1) +
      N(1, 1) * (std::pow(N(2, 1), 2)) * N(6, 1) +
      (std::pow(N(2, 1), 2)) * N(3, 1) * N(4, 1) -
      N(0, 1) * (std::pow(N(3, 1), 2)) * N(7, 1) -
      N(1, 1) * (std::pow(N(3, 1), 2)) * N(6, 1) -
      N(2, 1) * (std::pow(N(3, 1), 2)) * N(5, 1) +
      N(0, 1) * (std::pow(N(4, 1), 2)) * N(7, 1) +
      N(1, 1) * (std::pow(N(4, 1), 2)) * N(6, 1) +
      N(2, 1) * (std::pow(N(4, 1), 2)) * N(5, 1) -
      N(0, 1) * (std::pow(N(5, 1), 2)) * N(7, 1) -
      N(1, 1) * (std::pow(N(5, 1), 2)) * N(6, 1) -
      N(3, 1) * N(4, 1) * (std::pow(N(5, 1), 2)) -
      N(0, 1) * (std::pow(N(6, 1), 2)) * N(7, 1) -
      N(2, 1) * N(5, 1) * (std::pow(N(6, 1), 2)) -
      N(3, 1) * N(4, 1) * (std::pow(N(6, 1), 2)) +
      N(1, 1) * N(6, 1) * (std::pow(N(7, 1), 2)) +
      N(2, 1) * N(5, 1) * (std::pow(N(7, 1), 2)) +
      N(3, 1) * N(4, 1) * (std::pow(N(7, 1), 2)) +
      2 * N(0, 1) * N(1, 1) * N(2, 1) * N(4, 1) -
      2 * N(0, 1) * N(1, 1) * N(3, 1) * N(5, 1) -
      2 * N(0, 1) * N(2, 1) * N(3, 1) * N(6, 1) +
      2 * N(1, 1) * N(2, 1) * N(3, 1) * N(7, 1) -
      2 * N(0, 1) * N(4, 1) * N(5, 1) * N(6, 1) +
      2 * N(1, 1) * N(4, 1) * N(5, 1) * N(7, 1) +
      2 * N(2, 1) * N(4, 1) * N(6, 1) * N(7, 1) -
      2 * N(3, 1) * N(5, 1) * N(6, 1) * N(7, 1);
}

int SolveCalibRadialTrifocalTensor(const Eigen::Matrix<double, 2, 6>& x1,
                                   const Eigen::Matrix<double, 2, 6>& x2,
                                   const Eigen::Matrix<double, 2, 6>& x3,
                                   Eigen::Matrix<double, 8, 4>* sols) {
  // Setup nullspace
  Eigen::Matrix<double, 8, 6> A;
  for (int k = 0; k < 6; ++k) {
    double x1_1 = x1(0, k), x1_2 = x1(1, k);
    double x2_1 = x2(0, k), x2_2 = x2(1, k);
    double x3_1 = x3(0, k), x3_2 = x3(1, k);
    A.col(k) << x1_1 * x2_1 * x3_1, x1_2 * x2_1 * x3_1, x1_1 * x2_2 * x3_1,
        x1_2 * x2_2 * x3_1, x1_1 * x2_1 * x3_2, x1_2 * x2_1 * x3_2,
        x1_1 * x2_2 * x3_2, x1_2 * x2_2 * x3_2;
  }

  Eigen::Matrix<double, 8, 8> Q = A.colPivHouseholderQr().householderQ();
  Eigen::Matrix<double, 8, 2> N = Q.rightCols(2);
  Eigen::Array<double, 5, 1> c;
  EvaluateCalibTrifocalConstraintOnNullspace(N, &c);
  c = c / c(0);

  double roots[4];
  int n_sols = SolveQuarticReal(c(1), c(2), c(3), c(4), roots);

  for (int i = 0; i < n_sols; ++i) {
    sols->col(i) = N.col(0) * roots[i] + N.col(1);
  }

  return n_sols;
}




void EvaluateCalibMixedConstraintOnNullspace(const Eigen::Matrix<double, 12, 3> &N, Eigen::Array<double, 45, 1> *c) {

	double a[45];
	double b[45];
	
	double p[3];
	c->setZero();

	int mk;

	for (int k = 0; k < calib_mixed_coeffs_sz; k++) {

		mk = calib_mixed_coeffs_mm[8 * k];

		a[0] = N(mk, 0); a[1] = N(mk, 1); a[2] = N(mk, 2);

		mk = calib_mixed_coeffs_mm[8 * k + 1];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul1(a, p, b);

		mk = calib_mixed_coeffs_mm[8 * k + 2];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul2(b, p, a);

		mk = calib_mixed_coeffs_mm[8 * k + 3];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul3(a, p, b);

		mk = calib_mixed_coeffs_mm[8 * k + 4];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul4(b, p, a);

		mk = calib_mixed_coeffs_mm[8 * k + 5];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul5(a, p, b);

		mk = calib_mixed_coeffs_mm[8 * k + 6];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul6(b, p, a);

		mk = calib_mixed_coeffs_mm[8 * k + 7];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul7(a, p, b);

		for (int i = 0; i < 45; ++i) {
			(*c)(i) += calib_mixed_coeffs_cc[k] * b[i];
		}
	}
}


void EvaluateProjectiveMixedConstraintOnNullspace(const Eigen::Matrix<double, 12, 3> &N, Eigen::Array<double, 28, 1> *c) {

	double a[28];
	double b[28];

	double p[3];
	c->setZero();

	int mk;

	for (int k = 0; k < projective_mixed_coeffs_sz; k++) {

		mk = projective_mixed_coeffs_mm[6 * k];

		a[0] = N(mk, 0); a[1] = N(mk, 1); a[2] = N(mk, 2);

		mk = projective_mixed_coeffs_mm[6 * k + 1];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul1(a, p, b);

		mk = projective_mixed_coeffs_mm[6 * k + 2];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul2(b, p, a);

		mk = projective_mixed_coeffs_mm[6 * k + 3];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul3(a, p, b);

		mk = projective_mixed_coeffs_mm[6 * k + 4];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul4(b, p, a);

		mk = projective_mixed_coeffs_mm[6 * k + 5];
		p[0] = N(mk, 0); p[1] = N(mk, 1); p[2] = N(mk, 2);
		mul5(a, p, b);

		for (int i = 0; i < 28; ++i) {
			(*c)(i) += projective_mixed_coeffs_cc[k] * b[i];
		}
	}
}




void MixedTrifocalFastEigenVectorSolver(double* eigv, int neig, const Eigen::Matrix<double, 48, 48> & AM,
	 Eigen::Matrix<double, 2, 48> * sols) {
	static const int ind[] = { 7,11,17,25,35,47 };
	// Truncated action matrix containing non-trivial rows
	Eigen::Matrix<double, 6, 48> AMs;
	double zi[13];

	for (int i = 0; i < 6; i++) {
		AMs.row(i) = AM.row(ind[i]);
	}
	for (int i = 0; i < neig; i++) {
		zi[0] = eigv[i];
		for (int j = 1; j < 13; j++) {
			zi[j] = zi[j - 1] * eigv[i];
		}
		Eigen::Matrix<double, 6, 6> AA;
		AA.col(0) = AMs.col(5) + zi[0] * AMs.col(6) + zi[1] * AMs.col(7);
		AA.col(1) = AMs.col(4) + zi[0] * AMs.col(8) + zi[1] * AMs.col(9) + zi[2] * AMs.col(10) + zi[3] * AMs.col(11);
		AA.col(2) = AMs.col(3) + zi[0] * AMs.col(12) + zi[1] * AMs.col(13) + zi[2] * AMs.col(14) + zi[3] * AMs.col(15) + zi[4] * AMs.col(16) + zi[5] * AMs.col(17);
		AA.col(3) = AMs.col(2) + zi[0] * AMs.col(18) + zi[1] * AMs.col(19) + zi[2] * AMs.col(20) + zi[3] * AMs.col(21) + zi[4] * AMs.col(22) + zi[5] * AMs.col(23) + zi[6] * AMs.col(24) + zi[7] * AMs.col(25);
		AA.col(4) = AMs.col(1) + zi[0] * AMs.col(26) + zi[1] * AMs.col(27) + zi[2] * AMs.col(28) + zi[3] * AMs.col(29) + zi[4] * AMs.col(30) + zi[5] * AMs.col(31) + zi[6] * AMs.col(32) + zi[7] * AMs.col(33) + zi[8] * AMs.col(34) + zi[9] * AMs.col(35);
		AA.col(5) = AMs.col(0) + zi[0] * AMs.col(36) + zi[1] * AMs.col(37) + zi[2] * AMs.col(38) + zi[3] * AMs.col(39) + zi[4] * AMs.col(40) + zi[5] * AMs.col(41) + zi[6] * AMs.col(42) + zi[7] * AMs.col(43) + zi[8] * AMs.col(44) + zi[9] * AMs.col(45) + zi[10] * AMs.col(46) + zi[11] * AMs.col(47);
		AA(0, 0) = AA(0, 0) - zi[2];
		AA(1, 1) = AA(1, 1) - zi[4];
		AA(2, 2) = AA(2, 2) - zi[6];
		AA(3, 3) = AA(3, 3) - zi[8];
		AA(4, 4) = AA(4, 4) - zi[10];
		AA(5, 5) = AA(5, 5) - zi[12];

		Eigen::Matrix<double, 5, 1>  s = AA.leftCols(5).colPivHouseholderQr().solve(-AA.col(5));
		(*sols)(0, i) = s(4);
		(*sols)(1, i) = zi[0];
	}
}


int SolverCalibMixedTrifocalTensorActionMatrix(const double *data, Eigen::Matrix<double, 2, 48> *sols)
{
	// Setup elimination template
	static const int coeffs0_ind[] = { 0,45,1,0,45,46,3,1,0,45,46,48,6,3,1,0,45,46,48,51,10,6,3,1,0,45,46,48,51,55,15,10,6,3,1,0,45,46,48,51,55,60,21,15,10,6,3,1,45,46,48,51,55,60,66,28,21,15,10,6,3,46,45,48,51,55,60,66,36,28,21,15,10,6,48,46,51,55,60,66,36,28,21,15,10,51,48,55,60,66,36,28,21,15,55,51,60,66,36,28,21,60,55,66,36,28,66,60,2,0,45,47,4,2,47,1,0,45,46,49,7,4,2,47,49,3,1,0,45,46,48,52,11,7,4,2,47,49,52,6,3,1,45,0,46,48,51,56,16,11,7,4,2,47,49,52,56,10,6,3,46,45,1,48,0,51,55,61,22,16,11,7,4,2,47,49,52,56,61,15,10,6,48,46,45,3,51,1,55,60,67,29,22,16,11,7,4,47,49,52,56,61,67,21,15,10,51,48,46,45,6,55,3,60,66,37,29,22,16,11,7,49,47,52,56,61,67,28,21,15,55,51,48,46,10,60,6,66,37,29,22,16,11,52,49,56,61,67,36,28,21,60,55,51,48,15,66,10,37,29,22,16,56,52,61,67,36,28,66,60,55,51,21,15,37,29,22,61,56,67,36,66,60,55,28,21,5,2,0,45,47,50,8,5,50,4,2,47,1,45,46,49,0,53,12,8,5,50,53,7,4,2,47,49,3,46,0,45,48,52,1,57,17,12,8,5,50,53,57,11,7,4,47,2,49,52,6,48,1,45,46,0,51,56,3,62,23,17,12,8,5,50,53,57,62,16,11,7,49,47,4,52,2,56,10,51,3,46,45,48,1,55,61,6,68,30,23,17,12,8,5,50,53,57,62,68,22,16,11,52,49,47,7,56,4,61,15,55,6,48,46,45,51,3,60,67,10,38,30,23,17,12,8,50,53,57,62,68,29,22,16,56,52,49,47,11,61,7,67,21,60,10,51,48,46,55,6,66,15,38,30,23,17,12,53,50,57,62,68,37,29,22,61,56,52,49,16,67,11,28,66,15,55,51,48,60,10,21,38,30,23,17,57,53,62,68,37,29,67,61,56,52,22,16,36,21,60,55,51,66,15,28,9,5,2,47,45,0,50,54,13,9,54,8,5,50,4,47,49,46,45,0,1,53,2,58,18,13,9,54,58,12,8,5,50,53,7,49,2,47,52,48,46,45,0,1,3,57,4,63,24,18,13,9,54,58,63,17,12,8,50,5,53,57,11,52,4,47,49,2,56,51,48,46,1,3,45,6,62,7,69,31,24,18,13,9,54,58,63,69,23,17,12,53,50,8,57,5,62,16,56,7,49,47,52,4,61,55,51,48,3,6,45,46,10,68,11,39,31,24,18,13,9,54,58,63,69,30,23,17,57,53,50,12,62,8,68,22,61,11,52,49,47,56,7,67,60,55,51,6,10,46,48,15,16,39,31,24,18,13,54,58,63,69,38,30,23,62,57,53,50,17,68,12,29,67,16,56,52,49,61,11,66,60,55,10,15,48,51,21,22,14,9,5,50,47,45,0,2,54,59,19,14,59,13,9,54,8,50,53,49,47,2,46,45,1,0,4,58,5,64,25,19,14,59,64,18,13,9,54,58,12,53,5,50,57,52,49,47,2,4,48,46,45,3,1,7,63,8,70,32,25,19,14,59,64,70,24,18,13,54,9,58,63,17,57,8,50,53,5,62,56,52,49,4,7,47,51,48,46,45,6,3,11,69,12,40,32,25,19,14,59,64,70,31,24,18,58,54,13,63,9,69,23,62,12,53,50,57,8,68,61,56,52,7,11,47,49,55,51,48,46,10,6,16,17,20,14,9,54,50,47,2,45,0,5,59,65,26,20,65,19,14,59,13,54,58,53,50,5,49,47,4,46,1,45,2,8,64,9,71,33,26,20,65,71,25,19,14,59,64,18,58,9,54,63,57,53,50,5,8,52,49,47,7,48,3,46,4,45,12,70,13,27,20,14,59,54,50,5,47,2,45,9,65,72,34,27,72,26,20,65,19,59,64,58,54,9,53,50,8,49,4,47,46,45,5,13,71,14,35,27,20,65,59,54,9,50,5,47,45,14,72,41,33,26,20,65,71,32,25,19,59,14,64,70,24,63,13,54,58,9,69,62,57,53,8,12,50,56,52,49,47,11,51,6,48,7,46,17,18,40,32,25,19,14,59,64,70,39,31,24,63,58,54,18,69,13,30,68,17,57,53,50,62,12,67,61,56,11,16,49,52,60,55,51,48,15,10,22,23,39,31,24,18,58,54,63,69,38,30,68,62,57,53,23,17,37,22,61,56,52,67,16,66,60,15,21,51,55,28,29,38,30,23,62,57,68,37,67,61,56,29,22,28,66,60,55,21,36,37,29,67,61,66,60,36,28,36,66 };
	static const int coeffs1_ind[] = { 44,72,44,35,72,65,35,72,27,65,59,44,44,72,27,65,20,59,54,35,44,35,72,65,20,59,14,54,50,27,44,35,27,72,65,59,14,54,9,50,47,20,43,35,34,27,72,26,65,71,64,59,14,58,54,13,53,8,50,49,47,46,9,19,20,42,34,27,72,33,26,20,65,71,25,64,14,59,70,63,58,54,9,13,57,53,50,12,52,7,49,48,46,8,47,18,19,44,43,35,34,72,71,65,20,64,59,19,58,13,54,53,50,49,14,26,27,43,35,42,34,27,72,33,71,20,65,70,64,59,14,19,63,58,54,18,57,12,53,52,49,48,13,50,25,26,42,34,27,72,41,33,26,65,20,71,32,70,19,59,64,14,69,63,58,13,18,54,62,57,53,50,17,56,11,52,51,48,12,49,24,25,41,33,26,20,65,71,40,32,25,64,59,19,70,14,31,69,18,58,54,63,13,68,62,57,12,17,50,53,61,56,52,49,16,55,10,51,11,48,23,24,44,43,72,27,71,65,26,64,19,59,58,54,53,20,34,35,44,43,35,42,27,72,71,65,20,26,70,64,59,25,63,18,58,57,53,52,19,54,33,34,43,35,42,34,72,27,41,26,65,71,20,70,64,19,25,59,69,63,58,54,24,62,17,57,56,52,51,18,53,32,33,42,34,27,72,41,33,71,65,26,20,40,25,64,59,70,19,69,63,18,24,54,58,68,62,57,53,23,61,16,56,55,51,17,52,31,32,41,33,26,20,65,71,40,32,70,64,59,25,19,39,24,63,58,54,69,18,68,62,17,23,53,57,67,61,56,52,22,60,15,55,16,51,30,31,40,32,25,19,59,64,70,39,31,69,63,58,54,24,18,38,23,62,57,53,68,17,67,61,16,22,52,56,66,60,55,51,21,15,29,30,35,72,34,71,26,65,64,59,58,27,43,44,44,35,72,27,34,71,65,33,70,25,64,63,58,57,26,59,42,43,44,43,35,34,72,27,71,26,33,65,70,64,59,32,69,24,63,62,57,56,25,58,41,42,43,35,42,72,34,27,33,71,65,26,70,25,32,59,64,69,63,58,31,68,23,62,61,56,55,24,57,40,41,42,34,27,72,41,71,65,33,26,32,70,64,59,25,69,24,31,58,63,68,62,57,30,67,22,61,60,55,23,56,39,40,41,33,26,65,71,40,70,64,59,32,25,31,69,63,58,24,68,23,30,57,62,67,61,56,29,66,21,60,22,55,38,39,40,32,25,64,59,70,39,69,63,58,31,24,30,68,62,57,23,67,22,29,56,61,66,60,55,28,21,37,38,39,31,24,63,58,69,38,68,62,57,30,23,29,67,61,56,22,66,21,28,55,60,36,37,44,43,34,72,71,65,64,35,44,35,43,72,42,33,71,70,64,63,34,65,44,43,35,34,42,72,71,65,41,32,70,69,63,62,33,64,44,43,35,42,72,34,33,41,65,71,70,64,40,31,69,68,62,61,32,63,43,35,72,42,34,41,71,65,33,32,40,64,70,69,63,39,30,68,67,61,60,31,62,42,34,72,71,65,41,33,40,70,64,32,31,39,63,69,68,62,38,29,67,66,60,30,61,41,33,71,65,70,64,40,32,39,69,63,31,30,38,62,68,67,61,37,28,66,29,60,40,32,70,64,69,63,39,31,38,68,62,30,29,37,61,67,66,60,36,28,39,31,69,63,68,62,38,30,37,67,61,29,28,36,60,66,38,30,68,62,67,61,37,29,36,66,60,28,43,72,71,44,44,42,71,70,43,72,44,43,72,41,70,69,42,71,44,43,42,72,71,40,69,68,41,70,44,43,72,42,41,71,70,39,68,67,40,69,43,72,42,71,41,40,70,69,38,67,66,39,68,42,72,71,41,70,40,39,69,68,37,66,38,67,41,71,70,40,69,39,38,68,67,36,37,66,40,70,69,39,68,38,37,67,66,36,39,69,68,38,67,37,36,66,38,68,67,37,66,36,37,67,66,36 };

	static const int C0_ind[] = { 0,56,57,58,69,113,114,115,116,125,126,170,171,172,173,174,181,182,183,227,228,229,230,231,232,237,238,239,240,284,285,286,287,288,289,290,293,294,295,296,297,341,342,343,344,345,346,347,348,350,351,352,353,354,398,399,400,401,402,403,404,405,406,407,408,409,410,411,456,457,458,459,460,461,462,463,464,465,466,467,514,515,516,517,518,519,520,521,522,523,572,573,574,575,576,577,578,579,630,631,632,633,634,635,688,689,690,691,741,754,795,797,798,799,810,811,812,821,852,854,855,856,857,866,867,868,869,870,876,878,909,911,912,913,914,915,922,923,924,925,926,927,928,932,933,935,966,968,969,970,971,972,973,978,979,980,981,982,983,984,985,986,989,990,991,992,1023,1025,1026,1027,1028,1029,1030,1031,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1046,1047,1048,1049,1080,1082,1083,1084,1085,1086,1087,1088,1089,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1137,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1256,1257,1258,1259,1260,1261,1262,1263,1268,1269,1270,1271,1272,1273,1274,1276,1314,1315,1316,1317,1318,1319,1326,1328,1329,1330,1331,1333,1368,1381,1392,1400,1422,1424,1425,1426,1437,1438,1439,1448,1449,1450,1457,1479,1480,1481,1482,1483,1484,1493,1494,1495,1496,1497,1503,1505,1506,1507,1508,1512,1514,1536,1537,1538,1539,1540,1541,1542,1549,1550,1551,1552,1553,1554,1555,1559,1560,1562,1563,1564,1565,1566,1569,1570,1571,1593,1594,1595,1596,1597,1598,1599,1600,1605,1606,1607,1608,1609,1610,1611,1612,1613,1616,1617,1618,1619,1620,1621,1622,1623,1624,1626,1627,1628,1650,1651,1652,1653,1654,1655,1656,1657,1658,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1707,1708,1710,1711,1712,1713,1714,1715,1716,1718,1719,1720,1721,1723,1724,1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1765,1768,1769,1770,1771,1772,1773,1774,1775,1776,1777,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1791,1792,1793,1794,1795,1796,1797,1798,1822,1826,1827,1828,1829,1830,1831,1832,1833,1838,1839,1840,1841,1842,1843,1844,1846,1848,1850,1851,1852,1853,1854,1855,1879,1881,1894,1905,1913,1914,1934,1935,1937,1938,1939,1950,1951,1952,1961,1962,1963,1970,1971,1972,1975,1991,1992,1993,1994,1995,1996,1997,2006,2007,2008,2009,2010,2016,2018,2019,2020,2021,2025,2027,2028,2029,2030,2031,2032,2048,2049,2050,2051,2052,2053,2054,2055,2062,2063,2064,2065,2066,2067,2068,2072,2073,2075,2076,2077,2078,2079,2082,2083,2084,2085,2086,2087,2088,2089,2091,2105,2106,2107,2108,2109,2110,2111,2112,2113,2118,2119,2120,2121,2122,2123,2124,2125,2126,2129,2130,2131,2132,2133,2134,2135,2136,2137,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2162,2163,2164,2166,2167,2168,2169,2170,2171,2174,2175,2176,2177,2179,2180,2181,2182,2183,2184,2186,2187,2188,2189,2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2219,2221,2224,2225,2226,2227,2228,2229,2231,2232,2233,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2247,2248,2249,2250,2251,2252,2253,2254,2256,2257,2258,2259,2260,2261,2262,2276,2278,2280,2293,2304,2312,2313,2320,2324,2333,2334,2336,2337,2338,2349,2350,2351,2360,2361,2362,2369,2370,2371,2374,2377,2378,2381,2388,2390,2391,2392,2393,2394,2395,2396,2405,2406,2407,2408,2409,2415,2417,2418,2419,2420,2424,2426,2427,2428,2429,2430,2431,2434,2435,2436,2438,2445,2447,2448,2449,2450,2451,2452,2453,2454,2461,2462,2463,2464,2465,2466,2467,2471,2472,2474,2475,2476,2477,2478,2481,2482,2483,2484,2485,2486,2487,2488,2490,2491,2492,2493,2494,2495,2502,2504,2505,2506,2508,2509,2510,2511,2512,2517,2518,2519,2521,2522,2523,2524,2525,2528,2529,2530,2531,2532,2533,2534,2535,2536,2538,2539,2540,2541,2542,2543,2544,2545,2546,2547,2548,2549,2550,2551,2552,2559,2561,2563,2565,2578,2589,2597,2598,2605,2609,2610,2611,2618,2619,2621,2622,2623,2634,2635,2636,2645,2646,2647,2654,2655,2656,2659,2662,2663,2666,2667,2668,2669,2673,2675,2676,2677,2678,2679,2680,2681,2690,2691,2692,2693,2694,2700,2702,2703,2704,2705,2709,2711,2712,2713,2714,2715,2716,2719,2720,2721,2723,2724,2725,2726,2730,2731,2732,2733,2734,2736,2749,2760,2768,2769,2776,2780,2781,2782,2784,2789,2790,2792,2793,2794,2805,2806,2807,2816,2817,2818,2825,2826,2827,2830,2833,2834,2837,2838,2839,2840,2841,2842,2844,2846,2847,2848,2850,2863,2874,2882,2883,2890,2894,2895,2896,2898,2900,2903,2904,2907,2908,2909,2910,2917,2918,2920,2921,2922,2923,2927,2928,2930,2931,2932,2933,2934,2937,2938,2939,2940,2941,2942,2943,2944,2946,2947,2948,2949,2950,2951,2952,2953,2954,2958,2959,2960,2962,2965,2966,2967,2968,2969,2972,2973,2974,2977,2978,2979,2980,2981,2982,2984,2985,2986,2988,2989,2990,2991,2992,2993,2994,2995,2997,2998,2999,3000,3001,3002,3003,3004,3005,3006,3007,3008,3015,3017,3019,3023,3024,3025,3026,3027,3028,3029,3030,3035,3036,3037,3038,3039,3040,3041,3043,3045,3047,3048,3049,3050,3051,3052,3055,3056,3057,3058,3059,3060,3074,3076,3081,3082,3083,3084,3085,3086,3093,3095,3096,3097,3098,3100,3104,3105,3106,3107,3109,3133,3139,3140,3141,3142,3153,3154,3155,3157,3197,3199 };
	static const int C1_ind[] = { 46,50,101,103,105,107,158,159,160,162,164,167,195,211,215,216,217,219,221,224,241,252,261,268,272,273,274,276,278,281,285,298,309,317,318,325,329,330,331,333,335,338,342,343,355,356,365,366,367,374,375,376,379,382,383,386,387,388,389,390,391,392,393,395,397,399,400,401,410,412,413,414,420,422,423,424,425,429,431,432,433,434,435,436,439,440,441,443,444,445,446,447,448,450,451,452,454,457,469,470,480,481,489,490,493,496,497,500,501,502,503,504,505,506,507,509,511,514,515,526,527,528,534,537,538,539,543,546,547,548,549,550,553,554,555,557,558,559,560,561,562,563,564,565,566,568,571,572,573,580,583,584,585,586,590,591,594,595,596,597,600,601,603,604,605,606,607,609,610,611,612,613,614,615,616,617,618,619,621,622,623,625,628,629,630,631,636,637,640,641,642,643,644,647,648,649,651,652,653,654,655,657,658,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,678,679,680,682,698,708,718,721,724,725,728,729,730,731,732,733,734,735,737,739,743,755,756,765,767,771,775,776,777,778,781,782,783,785,786,787,788,789,790,791,792,793,794,796,800,801,812,813,814,818,822,824,825,828,829,832,833,834,835,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,853,857,858,859,864,869,870,871,872,875,877,879,881,882,883,885,886,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,906,907,908,910,914,915,916,917,920,921,926,927,928,929,930,932,934,936,938,939,940,941,942,943,946,947,948,949,950,951,952,953,954,955,956,957,958,959,963,964,965,967,971,972,973,974,975,977,978,983,984,985,986,987,988,989,991,993,995,996,997,998,999,1000,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1020,1022,1024,1063,1067,1070,1071,1072,1073,1074,1075,1076,1077,1079,1081,1098,1109,1118,1119,1120,1124,1125,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1138,1143,1155,1160,1166,1167,1171,1175,1176,1177,1179,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1195,1200,1201,1212,1214,1217,1219,1223,1224,1225,1228,1232,1233,1234,1235,1236,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1252,1257,1258,1259,1262,1269,1271,1272,1274,1276,1280,1281,1282,1283,1285,1289,1290,1291,1292,1293,1295,1296,1297,1298,1299,1300,1301,1302,1303,1305,1306,1307,1309,1314,1315,1316,1317,1319,1326,1328,1329,1330,1331,1333,1337,1338,1339,1340,1342,1346,1347,1348,1349,1350,1352,1353,1354,1355,1356,1357,1358,1362,1363,1364,1366,1371,1372,1373,1374,1375,1376,1383,1385,1386,1387,1388,1390,1394,1395,1396,1397,1399,1403,1404,1405,1406,1407,1409,1410,1411,1412,1419,1421,1423,1428,1429,1430,1431,1432,1433,1440,1442,1443,1444,1445,1447,1451,1452,1453,1454,1456,1460,1461,1462,1463,1464,1478,1480,1519,1526,1528,1529,1530,1531,1532,1533,1565,1575,1576,1581,1583,1585,1586,1587,1588,1589,1590,1591,1616,1622,1627,1632,1633,1635,1638,1639,1640,1642,1643,1644,1645,1646,1647,1648,1657,1673,1675,1679,1681,1684,1689,1690,1691,1692,1695,1696,1697,1699,1700,1701,1702,1703,1704,1705,1714,1715,1728,1730,1732,1736,1738,1739,1741,1746,1747,1748,1749,1752,1753,1754,1756,1757,1758,1759,1760,1761,1762,1771,1772,1773,1785,1786,1787,1789,1793,1795,1796,1798,1803,1804,1805,1806,1809,1810,1811,1813,1814,1815,1816,1818,1819,1828,1829,1830,1831,1842,1843,1844,1846,1850,1852,1853,1855,1860,1861,1862,1863,1866,1867,1868,1870,1871,1875,1876,1885,1886,1887,1888,1899,1900,1901,1903,1907,1909,1910,1912,1917,1918,1919,1920,1923,1924,1925,1932,1942,1943,1944,1945,1956,1957,1958,1960,1964,1966,1967,1969,1974,1975,1976,1977,1999,2000,2001,2002,2013,2014,2015,2017,2021,2023,2024,2026,2098,2101,2102,2103,2145,2155,2158,2159,2160,2161,2197,2202,2209,2212,2215,2216,2217,2218,2245,2254,2259,2261,2266,2269,2272,2273,2274,2275,2285,2302,2309,2311,2316,2318,2323,2326,2329,2330,2331,2332,2342,2356,2359,2366,2368,2373,2375,2380,2383,2386,2387,2388,2389,2399,2401,2413,2416,2423,2425,2430,2432,2437,2440,2443,2445,2446,2456,2458,2470,2473,2480,2482,2487,2489,2494,2497,2502,2503,2513,2515,2527,2530,2537,2539,2544,2546,2551,2559,2570,2572,2584,2587,2594,2596,2601,2603,2627,2629,2641,2644,2651,2653,2684,2686,2698,2701 };

	Eigen::Matrix<double, 57, 57> C0;
	Eigen::Matrix<double, 57, 48> C1;
	C0.setZero();
	C1.setZero();
	for (int i = 0; i < 1070; i++) { C0(C0_ind[i]) = data[coeffs0_ind[i]]; }
	for (int i = 0; i < 883; i++) { C1(C1_ind[i]) = data[coeffs1_ind[i]]; }
	Eigen::Matrix<double, 57, 48> C12 = C0.partialPivLu().solve(C1);

	// Setup action matrix
	Eigen::Matrix<double, 54, 48> RR;
	RR << -C12.bottomRows(6), Eigen::Matrix<double, 48, 48>::Identity(48, 48);

	static const int AM_ind[] = { 42,32,24,18,14,12,13,0,15,16,17,1,19,20,21,22,23,2,25,26,27,28,29,30,31,3,33,34,35,36,37,38,39,40,41,4,43,44,45,46,47,48,49,50,51,52,53,5 };
	Eigen::Matrix<double, 48, 48> AM;
	for (int i = 0; i < 48; i++) {
		AM.row(i) = RR.row(AM_ind[i]);
	}

	// Solve eigenvalue problem

	Eigen::EigenSolver<Eigen::Matrix<double, 48, 48>> es(AM, false);
	if(es.info() != Eigen::ComputationInfo::Success)
		return 0;

	Eigen::Array<std::complex<double>, 48, 1> D = es.eigenvalues();
	

	int nroots = 0;
	double eigv[48];
	for (int i = 0; i < 48; i++) {
		if (std::abs(D(i).imag()) < 1e-6)
			eigv[nroots++] = D(i).real();
	}
	
	MixedTrifocalFastEigenVectorSolver(eigv, nroots, AM, sols);

	return nroots;
}

void CalibMixedTrifocalTensorRootRefinement(const Eigen::Array<double, 45, 1> &cc,
	const Eigen::Array<double, 28, 1> &cp,
	Eigen::Matrix<double,2,48> *sols, int n_sols)
{
	Eigen::Matrix<double, 2, 2> J;
	Eigen::Matrix<double, 2, 1> r;
	Eigen::Matrix<double, 45, 1> mm;

	for(int i = 0; i < n_sols; ++i) {

		for(int iter = 0; iter < 10; ++iter) {
			double x = (*sols)(0,i);
			double y = (*sols)(1,i);

			double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x, x6 = x5 * x, x7 = x6 * x, x8 = x7 * x;
			double y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y, y6 = y5 * y, y7 = y6 * y, y8 = y7 * y;
			
			mm << x8,x7*y,x7,x6*y2,x6*y,x6,x5*y3,x5*y2,x5*y,x5,x4*y4,x4*y3,x4*y2,x4*y,x4,x3*y5,x3*y4,x3*y3,x3*y2,x3*y,x3,x2*y6,x2*y5,x2*y4,x2*y3,x2*y2,x2*y,x2,x*y7,x*y6,x*y5,x*y4,x*y3,x*y2,x*y,x,y8,y7,y6,y5,y4,y3,y2,y,1;

			r(0) = cc(0)*mm(0)+cc(1)*mm(1)+cc(2)*mm(2)+cc(3)*mm(3)+cc(4)*mm(4)+cc(5)*mm(5)+cc(6)*mm(6)+cc(7)*mm(7)+cc(8)*mm(8)+cc(9)*mm(9)+cc(10)*mm(10)+cc(11)*mm(11)+cc(12)*mm(12)+cc(13)*mm(13)+cc(14)*mm(14)+cc(15)*mm(15)+cc(16)*mm(16)+cc(17)*mm(17)+cc(18)*mm(18)+cc(19)*mm(19)+cc(20)*mm(20)+cc(21)*mm(21)+cc(22)*mm(22)+cc(23)*mm(23)+cc(24)*mm(24)+cc(25)*mm(25)+cc(26)*mm(26)+cc(27)*mm(27)+cc(28)*mm(28)+cc(29)*mm(29)+cc(30)*mm(30)+cc(31)*mm(31)+cc(32)*mm(32)+cc(33)*mm(33)+cc(34)*mm(34)+cc(35)*mm(35)+cc(36)*mm(36)+cc(37)*mm(37)+cc(38)*mm(38)+cc(39)*mm(39)+cc(40)*mm(40)+cc(41)*mm(41)+cc(42)*mm(42)+cc(43)*mm(43)+cc(44)*mm(44);
			r(1) = cp(0)*mm(5)+cp(1)*mm(8)+cp(2)*mm(9)+cp(3)*mm(12)+cp(4)*mm(13)+cp(5)*mm(14)+cp(6)*mm(17)+cp(7)*mm(18)+cp(8)*mm(19)+cp(9)*mm(20)+cp(10)*mm(23)+cp(11)*mm(24)+cp(12)*mm(25)+cp(13)*mm(26)+cp(14)*mm(27)+cp(15)*mm(30)+cp(16)*mm(31)+cp(17)*mm(32)+cp(18)*mm(33)+cp(19)*mm(34)+cp(20)*mm(35)+cp(21)*mm(38)+cp(22)*mm(39)+cp(23)*mm(40)+cp(24)*mm(41)+cp(25)*mm(42)+cp(26)*mm(43)+cp(27)*mm(44);

			if(r.norm() < 1e-12)
				break;

			J(0,0) = 8*cc(0)*mm(2)+7*cc(1)*mm(4)+7*cc(2)*mm(5)+6*cc(3)*mm(7)+6*cc(4)*mm(8)+6*cc(5)*mm(9)+5*cc(6)*mm(11)+5*cc(7)*mm(12)+5*cc(8)*mm(13)+5*cc(9)*mm(14)+4*cc(10)*mm(16)+4*cc(11)*mm(17)+4*cc(12)*mm(18)+4*cc(13)*mm(19)+4*cc(14)*mm(20)+3*cc(15)*mm(22)+3*cc(16)*mm(23)+3*cc(17)*mm(24)+3*cc(18)*mm(25)+3*cc(19)*mm(26)+3*cc(20)*mm(27)+2*cc(21)*mm(29)+2*cc(22)*mm(30)+2*cc(23)*mm(31)+2*cc(24)*mm(32)+2*cc(25)*mm(33)+2*cc(26)*mm(34)+2*cc(27)*mm(35)+cc(28)*mm(37)+cc(29)*mm(38)+cc(30)*mm(39)+cc(31)*mm(40)+cc(32)*mm(41)+cc(33)*mm(42)+cc(34)*mm(43)+cc(35)*mm(44);
			J(1,0) = 6*cp(0)*mm(9)+5*cp(1)*mm(13)+5*cp(2)*mm(14)+4*cp(3)*mm(18)+4*cp(4)*mm(19)+4*cp(5)*mm(20)+3*cp(6)*mm(24)+3*cp(7)*mm(25)+3*cp(8)*mm(26)+3*cp(9)*mm(27)+2*cp(10)*mm(31)+2*cp(11)*mm(32)+2*cp(12)*mm(33)+2*cp(13)*mm(34)+2*cp(14)*mm(35)+cp(15)*mm(39)+cp(16)*mm(40)+cp(17)*mm(41)+cp(18)*mm(42)+cp(19)*mm(43)+cp(20)*mm(44);
			J(0,1) = cc(1)*mm(2)+2*cc(3)*mm(4)+cc(4)*mm(5)+3*cc(6)*mm(7)+2*cc(7)*mm(8)+cc(8)*mm(9)+4*cc(10)*mm(11)+3*cc(11)*mm(12)+2*cc(12)*mm(13)+cc(13)*mm(14)+5*cc(15)*mm(16)+4*cc(16)*mm(17)+3*cc(17)*mm(18)+2*cc(18)*mm(19)+cc(19)*mm(20)+6*cc(21)*mm(22)+5*cc(22)*mm(23)+4*cc(23)*mm(24)+3*cc(24)*mm(25)+2*cc(25)*mm(26)+cc(26)*mm(27)+7*cc(28)*mm(29)+6*cc(29)*mm(30)+5*cc(30)*mm(31)+4*cc(31)*mm(32)+3*cc(32)*mm(33)+2*cc(33)*mm(34)+cc(34)*mm(35)+8*cc(36)*mm(37)+7*cc(37)*mm(38)+6*cc(38)*mm(39)+5*cc(39)*mm(40)+4*cc(40)*mm(41)+3*cc(41)*mm(42)+2*cc(42)*mm(43)+cc(43)*mm(44);
			J(1,1) = cp(1)*mm(9)+2*cp(3)*mm(13)+cp(4)*mm(14)+3*cp(6)*mm(18)+2*cp(7)*mm(19)+cp(8)*mm(20)+4*cp(10)*mm(24)+3*cp(11)*mm(25)+2*cp(12)*mm(26)+cp(13)*mm(27)+5*cp(15)*mm(31)+4*cp(16)*mm(32)+3*cp(17)*mm(33)+2*cp(18)*mm(34)+cp(19)*mm(35)+6*cp(21)*mm(39)+5*cp(22)*mm(40)+4*cp(23)*mm(41)+3*cp(24)*mm(42)+2*cp(25)*mm(43)+cp(26)*mm(44);

			J(0,0) += 1e-8;
			J(1,1) += 1e-8;

			sols->col(i) = sols->col(i) - J.partialPivLu().solve(r);
		}		
	}
}

int SolveCalibMixedTrifocalTensor(const Eigen::Matrix<double, 3, 9> & x1,
		const Eigen::Matrix<double, 2, 9> & x2, const Eigen::Matrix<double, 2, 9> & x3,
		Eigen::Matrix<double, 12, 48> * sols) {

	// Setup nullspace
	Eigen::Matrix<double, 12, 9> A;
	for (int k = 0; k < 9; ++k) {
		double x1_1 = x1(0, k), x1_2 = x1(1, k), x1_3 = x1(2, k);
		double x2_1 = x2(0, k), x2_2 = x2(1, k);
		double x3_1 = x3(0, k), x3_2 = x3(1, k);
		A.col(k) << x1_1 * x2_1 * x3_1, x1_2* x2_1* x3_1, x1_3* x2_1* x3_1, x1_1* x2_2* x3_1, x1_2* x2_2* x3_1, x1_3* x2_2* x3_1, x1_1* x2_1* x3_2, x1_2* x2_1* x3_2, x1_3* x2_1* x3_2, x1_1* x2_2* x3_2, x1_2* x2_2* x3_2, x1_3* x2_2* x3_2;
	}

	Eigen::Matrix<double, 12, 12> Q = A.colPivHouseholderQr().householderQ();
	Eigen::Matrix<double, 12, 3> N = Q.rightCols(3);

	Eigen::Array<double, 45, 1> cc;
	Eigen::Array<double, 28, 1> cp;

	cc /= cc.abs().mean();
	cp /= cp.abs().mean();

	EvaluateCalibMixedConstraintOnNullspace(N, &cc);
	EvaluateProjectiveMixedConstraintOnNullspace(N, &cp);

	double data[73];
	for (int i = 0; i < 45; ++i)
		data[i] = cc[i];
	for (int i = 0; i < 28; ++i)
		data[45 + i] = cp[i];

	Eigen::Matrix<double, 2, 48> sols_xy;

	int n_sols = SolverCalibMixedTrifocalTensorActionMatrix(data, &sols_xy);

	CalibMixedTrifocalTensorRootRefinement(cc, cp, &sols_xy, n_sols);

	for(int i = 0; i < n_sols; ++i) {
		sols->col(i) = sols_xy(0,i) * N.col(0) + sols_xy(1,i) * N.col(1) + N.col(2);
	}
	return n_sols;
}


}  // namespace init
}  // namespace colmap