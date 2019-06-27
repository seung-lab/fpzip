#ifndef PC_MAP_H
#define PC_MAP_H

#include <climits>
#if !defined WITH_REINTERPRET_CAST && !defined WITH_UNION
#include <cstring>
#endif

#define bitsizeof(t) ((unsigned)(CHAR_BIT * sizeof(t)))

template <typename T, unsigned width = bitsizeof(T), typename U = void>
struct PCmap;

// specialized for integer-to-integer map
template <typename T, unsigned width>
struct PCmap<T, width, void> {
  typedef T DOMAIN_TYPE;
  typedef T RANGE_TYPE;
  static const unsigned bits = width;                    // RANGE_TYPE bits
  static const unsigned shift = bitsizeof(RANGE_TYPE) - bits; // DOMAIN_TYPE\RANGE_TYPE bits
  RANGE_TYPE forward(DOMAIN_TYPE d) const { return d >> shift; }
  DOMAIN_TYPE inverse(RANGE_TYPE r) const { return r << shift; }
  DOMAIN_TYPE identity(DOMAIN_TYPE d) const { return inverse(forward(d)); }
};

// specialized for float type
template <unsigned width>
struct PCmap<float, width, void> {
  typedef float    DOMAIN_TYPE;
  typedef unsigned RANGE_TYPE;
  union UNION {
    UNION(DOMAIN_TYPE d) : d(d) {}
    UNION(RANGE_TYPE r) : r(r) {}
    DOMAIN_TYPE d;
    RANGE_TYPE r;
  };
  static const unsigned bits = width;                    // RANGE_TYPE bits
  static const unsigned shift = bitsizeof(RANGE_TYPE) - bits; // DOMAIN_TYPE\RANGE_TYPE bits
  RANGE_TYPE fcast(DOMAIN_TYPE d) const;
  DOMAIN_TYPE icast(RANGE_TYPE r) const;
  RANGE_TYPE forward(DOMAIN_TYPE d) const;
  DOMAIN_TYPE inverse(RANGE_TYPE r) const;
  DOMAIN_TYPE identity(DOMAIN_TYPE d) const;
};

// specialized for double type
template <unsigned width>
struct PCmap<double, width, void> {
  typedef double             DOMAIN_TYPE;
  typedef unsigned long long RANGE_TYPE;
  union UNION {
    UNION(DOMAIN_TYPE d) : d(d) {}
    UNION(RANGE_TYPE r) : r(r) {}
    DOMAIN_TYPE d;
    RANGE_TYPE r;
  };
  static const unsigned bits = width;                    // RANGE_TYPE bits
  static const unsigned shift = bitsizeof(RANGE_TYPE) - bits; // DOMAIN_TYPE\RANGE_TYPE bits
  RANGE_TYPE fcast(DOMAIN_TYPE d) const;
  DOMAIN_TYPE icast(RANGE_TYPE r) const;
  RANGE_TYPE forward(DOMAIN_TYPE d) const;
  DOMAIN_TYPE inverse(RANGE_TYPE r) const;
  DOMAIN_TYPE identity(DOMAIN_TYPE d) const;
};

#include "pcmap.inl"

#endif
