// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#pragma once
#include <assert.h>
#include <string>
#include <memory>

#define TT_THROW(...)                                         \
  do {                                                        \
    throw core::details::EnforceNotMet(__VA_ARGS__);          \
  } while (false)

#if !defined(_WIN32)
#define TT_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define TT_UNLIKELY(condition) (condition)
#endif

#define TT_ENFORCE(cond, ...)                                            \
  do {                                                                   \
    if (TT_UNLIKELY(!(cond))) {                                                         \
      std::string err_msg("enforce error");                              \
      err_msg += #cond;                                                  \
      err_msg += core::details::string_format(" at %s:%d\n", __FILE__, __LINE__);     \
      err_msg += core::details::string_format(__VA_ARGS__);                           \
      throw core::details::EnforceNotMet(err_msg);                                    \
    }                                                                    \
  } while (false)

#define TT_ENFORCE_EQ(a, b, ...) TT_ENFORCE((a) == (b), __VA_ARGS__)
#define TT_ENFORCE_NE(a, b, ...) TT_ENFORCE((a) != (b), __VA_ARGS__)
#define TT_ENFORCE_LT(a, b, ...) TT_ENFORCE((a) < (b), __VA_ARGS__)
#define TT_ENFORCE_LE(a, b, ...) TT_ENFORCE((a) <= (b), __VA_ARGS__)
#define TT_ENFORCE_GT(a, b, ...) TT_ENFORCE((a) > (b), __VA_ARGS__)
#define TT_ENFORCE_GE(a, b, ...) TT_ENFORCE((a) >= (b), __VA_ARGS__)

namespace ps_tensor {
namespace core {
namespace details {

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

class EnforceNotMet : public std::exception {
 public:
  explicit EnforceNotMet(std::string msg) : msg_(std::move(msg)) {
  }

  const char *what() const noexcept override {
    return msg_.c_str();
  }

 private:
  mutable std::string msg_;
};

}
}
}
