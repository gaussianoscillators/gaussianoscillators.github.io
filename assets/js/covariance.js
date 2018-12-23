// Copyright 2015 Charles R. Hogg III and Google Inc. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.// Utility functions for animated plots.

// Construct a covariance matrix for the points x from the covariance function
// kFunc.
function CovarianceMatrix(x, kFunc, noise_sigma) {
  noise_sigma = typeof noise_sigma !== 'undefined' ? noise_sigma : 0.0001;
  var noise = Independent(noise_sigma);
  return jStat.create(x.length, x.length, function(i, j) {
    return kFunc(x[i], x[j]) + noise(x[i], x[j]);
  });
};

////////////////////////////////////////////////////////////////////////////////
// Covariance functions.

// Independent points.
function Independent(sigma) {
  var sigma_squared = sigma * sigma;
  return function(x1, x2) {
    return (x1 == x2) ? sigma_squared : 0.0;
  };
};

// The following functions return 
function SquaredExponential(ell, sigma) {
  var sigma_squared = sigma * sigma;
  return function(x1, x2) {
    return sigma_squared * Math.exp(-Math.pow((x1 - x2) / ell, 2));
  };
};

function Exponential(ell, sigma) {
  var sigma_squared = sigma * sigma;
  return function(x1, x2) {
    return sigma_squared * Math.exp(-Math.abs((x1 - x2) / ell));
  };
};

function Cosine(ell, sigma) {
  var sigma_squared = sigma * sigma;
  return function(x1, x2) {
    return sigma_squared * Math.cos(Math.PI * (x2 - x1) / ell);
  };
};

function Matern3v2(ell, sigma) {
  var sigma_squared = sigma * sigma;
  var scaling_factor = Math.sqrt(3) / ell;
  return function(x1, x2) {
    var diff = Math.abs(x1 - x2) * scaling_factor;
    return sigma_squared * (1 + diff) * Math.exp(-diff);
  }
}

function Matern5v2(ell, sigma) {
  var sigma_squared = sigma * sigma;
  var scaling_factor = Math.sqrt(5) / ell;
  return function(x1, x2) {
    var diff = Math.abs(x1 - x2) * scaling_factor;
    return sigma_squared * (1 + diff + diff * diff / 3) * Math.exp(-diff);
  }
}
