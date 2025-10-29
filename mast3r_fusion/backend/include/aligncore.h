#include <torch/extension.h>
#include <vector>

class AlignCoreCalib
{
public:
  AlignCoreCalib(){}
  ~AlignCoreCalib(){}
public:
  void init(
  torch::Tensor _Twc, torch::Tensor _Xs, torch::Tensor _Cs,
  torch::Tensor _K,
  torch::Tensor _ii, torch::Tensor _jj, 
  torch::Tensor _idx_ii2jj, torch::Tensor _valid_match,
  torch::Tensor _Q,
  const int _height, const int _width,
  const int _pixel_border,
  const float _z_eps,
  const float _sigma_pixel, const float _sigma_depth,
  const float _C_thresh,
  const float _Q_thresh,
  const int _max_iter,
  const float _delta_thresh, int _subpixel_factor, float _d_diff_thresh);
  void hessian(torch::Tensor H, torch::Tensor v);
  void hessian_pieces(torch::Tensor H, torch::Tensor v, torch::Tensor cost);
  torch::Tensor retract(torch::Tensor _dx);
  
public:
  torch::Tensor Twc;
  torch::Tensor Xs;
  torch::Tensor Cs;
  torch::Tensor K;
  torch::Tensor ii;
  torch::Tensor jj; 
  torch::Tensor idx_ii2jj;
  torch::Tensor valid_match;
  torch::Tensor Q;
  int height;
  int width;
  int pixel_border;
  float z_eps;
  float sigma_pixel;
  float sigma_depth;
  float C_thresh;
  float Q_thresh;
  int max_iter;
  float delta_thresh;
  torch::Tensor dx;
  torch::Tensor delta_norm;
  int subpixel_factor;
  float d_diff_thresh;

};
