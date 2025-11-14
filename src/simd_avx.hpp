#ifndef SIMD_AVX_HPP
#define SIMD_AVX_HPP

#include <immintrin.h>


/*
  implementation of SIMDs for Intel-CPUs with AVX support:
  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */


namespace ASC_HPC
{


   template<>
  class SIMD<mask64,2>
  {
    __m128i m_mask;
  public:

    SIMD (__m128i mask) : m_mask(mask) { };
    SIMD (__m128d mask) : m_mask(_mm_castpd_si128 (mask)) { ; }
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }
    
    SIMD<mask64, 1> lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> hi() const { return SIMD<mask64,1>((*this)[1]); }
    SIMD(double v0, double v1)
    {(*this)[0] = v0; 
      (*this)[1] = v1;
    } 
  };
 

  template<>
  class SIMD<mask64,4>
  {
    __m256i m_mask;
  public:
    SIMD (__m256i mask) : m_mask(mask) { }
    SIMD (__m256d mask) : m_mask(_mm256_castpd_si256(mask)) { }
    SIMD (SIMD<mask64,2> lo, SIMD<mask64,2> hi) 
      : m_mask(_mm256_set_epi64x(hi[1].val(), hi[0].val(), lo[1].val(), lo[0].val())) { }
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }
    SIMD<mask64, 2> lo() const { return SIMD<mask64,2>((*this)[0], (*this)[1]); }
    SIMD<mask64, 2> hi() const { return SIMD<mask64,2>((*this)[2], (*this)[3]); }
  };

  // AVX/SSE specialization for SIMD<double,2> using 128-bit SSE registers
  template<>
  class SIMD<double,2>
  {
    __m128d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double val) : m_val{_mm_set1_pd(val)} {};
    SIMD(__m128d val) : m_val{val} {};
    SIMD (double v0, double v1) : m_val{_mm_set_pd(v1,v0)} {  }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) : SIMD(v0.val(), v1.val()) { }
    SIMD (std::array<double,2> a) : SIMD(a[0],a[1]) { }
    SIMD (double const * p) { m_val = _mm_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,2> mask) {
      alignas(16) int64_t m[2] = { mask[0].val(), mask[1].val() };
      __m128i mm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(m));
      m_val = _mm_maskload_pd(p, mm);
    }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 1> lo() const { return SIMD<double,1>((*this)[0]); }
    SIMD<double, 1> hi() const { return SIMD<double,1>((*this)[1]); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64,2> mask) const {
      alignas(16) int64_t m[2] = { mask[0].val(), mask[1].val() };
      __m128i mm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(m));
      _mm_maskstore_pd(p, mm, m_val);
    }
  };



   template<>
 class SIMD<double,2>
  {
    __m128d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double val) : m_val{_mm_set1_pd(val)} {};
    SIMD(__m128d val) : m_val{val} {};
    SIMD (double v0, double v1) : m_val{_mm_set_pd(v1,v0)} {  }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) :  m_val{_mm_set_pd(v1[0],v0[0])} { }  
    SIMD (std::array<double,4> a) : SIMD(a[0],a[1]) { }
    SIMD (double const * p) { m_val = _mm_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,2> mask) { m_val = _mm_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 1> lo() const { return SIMD<double,1>((*this)[0]); }
    SIMD<double, 1> hi() const { return SIMD<double,1>((*this)[1]); }

    // better:
    // SIMD<double, 2> lo() const { return _mm256_extractf128_pd(m_val, 0); }
    // SIMD<double, 2> hi() const { return _mm256_extractf128_pd(m_val, 1); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm_storeu_pd(p, m_val); }
   // void store (double * p, SIMD<mask64,2> mask) const { _mm256_maskstore_pd(p, mask.val(), m_val); }
  };
   

  
  template<>
  class SIMD<double,4>
  {
    __m256d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double val) : m_val{_mm256_set1_pd(val)} {};
    SIMD(__m256d val) : m_val{val} {};
    SIMD (double v0, double v1, double v2, double v3) : m_val{_mm256_set_pd(v3,v2,v1,v0)} {  }
    SIMD (SIMD<double,2> v0, SIMD<double,2> v1) 
      : m_val(_mm256_insertf128_pd(_mm256_castpd128_pd256(v0.val()), v1.val(), 1)) { }
    SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    SIMD (double const * p) { m_val = _mm256_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,4> mask) {
      alignas(32) int64_t m[4] = { mask[0].val(), mask[1].val(), mask[2].val(), mask[3].val() };
      __m256i mm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m));
      m_val = _mm256_maskload_pd(p, mm);
    }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 2> lo() const { return SIMD<double,2>(_mm256_extractf128_pd(m_val, 0)); }
    SIMD<double, 2> hi() const { return SIMD<double,2>(_mm256_extractf128_pd(m_val, 1)); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm256_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64,4> mask) const {
      alignas(32) int64_t m[4] = { mask[0].val(), mask[1].val(), mask[2].val(), mask[3].val() };
      __m256i mm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m));
      _mm256_maskstore_pd(p, mm, m_val);
    }
  };
  



  
  template<>
  class SIMD<int64_t,4>
  {
    __m256i m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t val) : m_val{_mm256_set1_epi64x(val)} {};
    SIMD(__m256i val) : m_val{val} {};
    SIMD (int64_t v0, int64_t v1, int64_t v2, int64_t v3) : m_val{_mm256_set_epi64x(v3,v2,v1,v0) } { } 
    SIMD (SIMD<int64_t,2> v0, SIMD<int64_t,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // can do better !
    // SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    // SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    // SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    // const double * Ptr() const { return (double*)&val; }
    // SIMD<double, 2> Lo() const { return _mm256_extractf128_pd(val, 0); }
    // SIMD<double, 2> Hi() const { return _mm256_extractf128_pd(val, 1); }
    int64_t operator[](size_t i) const { return ((int64_t*)&m_val)[i]; }
  };
  


  template <int64_t first>
  class IndexSequence<int64_t, 4, first> : public SIMD<int64_t,4>
  {
  public:
    IndexSequence()
      : SIMD<int64_t,4> (first, first+1, first+2, first+3) { }
  };
  


  
  inline auto operator+ (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_add_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_sub_pd(a.val(), b.val())); }
  inline auto operator* (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_mul_pd(a.val(), b.val())); }
  inline auto operator* (double a, SIMD<double,4> b) { return SIMD<double,4>(a)*b; }

  inline auto operator+ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_add_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_sub_pd(a.val(), b.val())); }
  inline auto operator* (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_mul_pd(a.val(), b.val())); }
  inline auto operator* (double a, SIMD<double,2> b) { return SIMD<double,2>(a)*b; }
  
#ifdef __FMA__
  inline SIMD<double,4> fma (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
  { return _mm256_fmadd_pd (a.val(), b.val(), c.val()); }
#endif

  // Transpose function: takes 4 rows as input, writes 4 columns as output
  inline void transpose(SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
                        SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3)
  {
    __m256d r0 = a0.val();
    __m256d r1 = a1.val();
    __m256d r2 = a2.val();
    __m256d r3 = a3.val();

    // unpack/shuffle within 128-bit lanes
    __m256d t0 = _mm256_unpacklo_pd(r0, r1); // a0[0], a1[0], a0[2], a1[2]
    __m256d t1 = _mm256_unpackhi_pd(r0, r1); // a0[1], a1[1], a0[3], a1[3]
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);

    // combine low/high 128-bit lanes to form columns
    __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20); // low halves
    __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31); // high halves
    __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

    b0 = SIMD<double,4>(c0);
    b1 = SIMD<double,4>(c1);
    b2 = SIMD<double,4>(c2);
    b3 = SIMD<double,4>(c3);
  }

  // ---------------------- Min/Max operations ------------------------------
  inline auto min(SIMD<double,4> a, SIMD<double,4> b) { 
    return SIMD<double,4>(_mm256_min_pd(a.val(), b.val())); 
  }

  inline auto max(SIMD<double,4> a, SIMD<double,4> b) { 
    return SIMD<double,4>(_mm256_max_pd(a.val(), b.val())); 
  }

  inline auto min(SIMD<double,2> a, SIMD<double,2> b) { 
    return SIMD<double,2>(_mm_min_pd(a.val(), b.val())); 
  }

  inline auto max(SIMD<double,2> a, SIMD<double,2> b) { 
    return SIMD<double,2>(_mm_max_pd(a.val(), b.val())); 
  }

  // ---------------------- Bitonic Sort for SIMD<double,4> -----------------
  // Sorts 4 doubles in ascending order within a single SIMD register
  inline SIMD<double,4> bitonic_sort(SIMD<double,4> v) {
    __m256d x = v.val();
    
    // Step 1: Compare-swap pairs (0,1) and (2,3)
    // Shuffle to get [1,0,3,2]
    __m256d y = _mm256_shuffle_pd(x, x, 0b0101);
    __m256d lo1 = _mm256_min_pd(x, y);
    __m256d hi1 = _mm256_max_pd(x, y);
    // Blend back: take min/max from correct positions
    __m256d t1 = _mm256_blend_pd(lo1, hi1, 0b1010); // [min(0,1), max(0,1), min(2,3), max(2,3)]
    
    // Step 2: Compare-swap (0,3) and (1,2) - bitonic merge
    // Permute 128-bit lanes and reverse within high lane: [0,1,3,2]
    __m256d t2 = _mm256_permute2f128_pd(t1, t1, 0x01); // swap lanes -> [2,3,0,1]
    t2 = _mm256_shuffle_pd(t2, t2, 0b0101);            // shuffle -> [3,2,1,0]
    
    __m256d lo2 = _mm256_min_pd(t1, t2);
    __m256d hi2 = _mm256_max_pd(t1, t2);
    __m256d t3 = _mm256_blend_pd(lo2, hi2, 0b1100); // [min, min, max, max]
    
    // Step 3: Final compare-swap on pairs
    __m256d t4 = _mm256_shuffle_pd(t3, t3, 0b0101);
    __m256d lo3 = _mm256_min_pd(t3, t4);
    __m256d hi3 = _mm256_max_pd(t3, t4);
    __m256d result = _mm256_blend_pd(lo3, hi3, 0b1010);
    
    return SIMD<double,4>(result);
  }

  // Bitonic sort for SIMD<double,2> - simple min/max swap
  inline SIMD<double,2> bitonic_sort(SIMD<double,2> v) {
    __m128d x = v.val();
    __m128d y = _mm_shuffle_pd(x, x, 0b01); // swap [0,1] -> [1,0]
    __m128d lo = _mm_min_pd(x, y);
    __m128d hi = _mm_max_pd(x, y);
    __m128d result = _mm_blend_pd(lo, hi, 0b10); // [min, max]
    return SIMD<double,2>(result);
  }

  
  



// Transpose function: takes 4 rows as input, writes 4 columns as output
  inline void transpose(SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
                        SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3)
  {
    __m256d r0 = a0.val();
    __m256d r1 = a1.val();
    __m256d r2 = a2.val();
    __m256d r3 = a3.val();

    // unpack/shuffle within 128-bit lanes
    __m256d t0 = _mm256_unpacklo_pd(r0, r1); // a0[0], a1[0], a0[2], a1[2]
    __m256d t1 = _mm256_unpackhi_pd(r0, r1); // a0[1], a1[1], a0[3], a1[3]
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);

    // combine low/high 128-bit lanes to form columns
    __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20); // low halves
    __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31); // high halves
    __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

    b0 = SIMD<double,4>(c0);
    b1 = SIMD<double,4>(c1);
    b2 = SIMD<double,4>(c2);
    b3 = SIMD<double,4>(c3);

  }
}



#endif




              