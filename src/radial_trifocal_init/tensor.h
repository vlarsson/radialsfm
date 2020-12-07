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

void RadialTensorCoordinateChange(const Eigen::Matrix<double, 8, 1>& T,
                                  const Eigen::Matrix<double, 2, 2>& A1,
                                  const Eigen::Matrix<double, 2, 2>& A2,
                                  const Eigen::Matrix<double, 2, 2>& A3,
                                  Eigen::Matrix<double, 8, 1>* Tout);
void MixedTensorCoordinateChange(const Eigen::Matrix<double, 12, 1>& T,
                                 const Eigen::Matrix<double, 3, 3>& A1,
                                 const Eigen::Matrix<double, 2, 2>& A2,
                                 const Eigen::Matrix<double, 2, 2>& A3,
                                 Eigen::Matrix<double, 12, 1>* Tout);
int FactorizeRadialTensor(const Eigen::Matrix<double, 8, 1>& T,
                          Eigen::Matrix<double, 2, 3> P1[2],
                          Eigen::Matrix<double, 2, 3> P2[2],
                          Eigen::Matrix<double, 2, 3> P3[2]);
void FactorizeMixedTensor(const Eigen::Matrix<double, 12, 1>& T,
                          Eigen::Matrix<double, 3, 4>* P1,
                          Eigen::Matrix<double, 2, 4>* P2,
                          Eigen::Matrix<double, 2, 4>* P3);
bool MetricUpgradeRadial(const Eigen::Matrix<double, 2, 3>& P2,
                         const Eigen::Matrix<double, 2, 3>& P3,
                         Eigen::Matrix<double, 3, 3>* H);
bool MetricUpgradeMixed(const Eigen::Matrix<double, 2, 4>& P2,
                        const Eigen::Matrix<double, 2, 4>& P3,
                        Eigen::Matrix<double, 4, 4>* H);
int SolveCalibRadialTrifocalTensor(const Eigen::Matrix<double, 2, 6>& x1,
                                   const Eigen::Matrix<double, 2, 6>& x2,
                                   const Eigen::Matrix<double, 2, 6>& x3,
                                   Eigen::Matrix<double, 8, 4>* sols);
int SolveCalibMixedTrifocalTensor(const Eigen::Matrix<double, 3, 9> & x1,
		const Eigen::Matrix<double, 2, 9> & x2, const Eigen::Matrix<double, 2, 9> & x3,
		Eigen::Matrix<double, 12, 48> * sols);
}  // namespace init
}  // namespace colmap