#pragma once

#include <stdexcept>

namespace svo {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ValidationError : public Error {
 public:
  using Error::Error;
};

}  // namespace svo
