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

#include "base/polynomial.h"
#include "util/types.h"
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

}  // namespace init
}  // namespace colmap