/*******************************************************************************
 * This file is part of the "https://github.com/fgarcia0x0/PolluxMathUtils"
 * For conditions of distribution and use, see copyright notice in LICENSE
 * Copyright (C) 2021, by Felipe Garcia (felipegarcia1402@gmail.com)
 ******************************************************************************/

#ifndef POLLUX_MATH_VECTOR_HPP
#define POLLUX_MATH_VECTOR_HPP

#include <ostream>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <numeric>
#include <cassert>
#include <type_traits>
#include <cstring>
#include <algorithm>
#include <functional>

/*
 * std::fma is constant expression in:
 *  - since GCC 8.1 support constexpr fma
 *  - since Clang 12.0 support constexpr fma
 */ 

#ifndef POLLUX_MATH_CONSTEXPR
    #define GCC_MAJOR   (__GNUC__)
    #define GCC_MINOR   (__GNUC_MINOR__)
    #define CLANG_MAJOR (__clang_major__)

    #define CLANG_SIG ((__GNUC__) && (__clang__))
    #define GCC_SIG ((__GNUC__) && (!CLANG_SIG))

    #if (CLANG_SIG && (CLANG_MAJOR >= 12)) || \
         GCC_SIG && (GCC_MAJOR > 8 || (GCC_MAJOR == 8 && GCC_MINOR >= 1))
        #define POLLUX_MATH_CONSTEXPR constexpr
    #else
        #define POLLUX_MATH_CONSTEXPR inline
    #endif
#endif

#ifndef POLLUX_MATH_CONSTINIT
    #ifdef __cpp_constinit
        #define POLLUX_MATH_CONSTINIT constinit
    #else
        #define POLLUX_MATH_CONSTINIT constexpr inline
    #endif
#endif

#ifndef POLLUX_MATH_CONCEPT
    #ifdef __cpp_lib_concepts
        #define POLLUX_MATH_CONCEPT concept
    #else
        #define POLLUX_MATH_CONCEPT constexpr inline bool
    #endif
#endif

namespace pollux::math::detail
{
    template <typename T> 
    constexpr inline const char* name_of = "unknown";

    #define REGISTER_NAMEOF(type) \
        template<> constexpr inline const char* name_of<type> = #type

    // Typedef's Ints
    REGISTER_NAMEOF(int8_t);
    REGISTER_NAMEOF(int16_t);
    REGISTER_NAMEOF(int32_t);
    REGISTER_NAMEOF(int64_t);
    REGISTER_NAMEOF(uint8_t);
    REGISTER_NAMEOF(uint16_t);
    REGISTER_NAMEOF(uint32_t);
    REGISTER_NAMEOF(uint64_t);

    // Floating Point
    REGISTER_NAMEOF(float);
    REGISTER_NAMEOF(double);
    REGISTER_NAMEOF(long double);

    #ifdef __cpp_lib_bit_cast
        using std::bit_cast;
    #else
        template <typename To, typename From>
        [[nodiscard]]
        constexpr To bit_cast(const From& src) noexcept
        {
            static_assert(sizeof(To) == sizeof(From), "This implementation additionally requires source type and the destination type have same size");
            static_assert(std::is_trivially_copyable_v<From>, "This implementation additionally requires source type to be trivially copyable");
            static_assert(std::is_trivially_copyable_v<To>, "This implementation additionally requires destination type to be trivially copyable");
            static_assert(std::is_trivially_constructible_v<To>, "This implementation additionally requires destination type to be trivially constructible");
        
            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }
    #endif

    template <typename T, typename... U>
    POLLUX_MATH_CONCEPT is_any_v = std::disjunction_v<std::is_same<T, U>...>;

    template <typename T, typename... Rest>
    POLLUX_MATH_CONCEPT all_same_v = std::conjunction_v<std::is_same<T, Rest>...>;

    template <typename T, typename U>
    POLLUX_MATH_CONCEPT is_same_v = std::is_same_v<T, U>;

    template <typename T, typename... Rest>
    POLLUX_MATH_CONCEPT all_convertible_v = std::conjunction_v<std::is_convertible<T, Rest>...>;

	template <typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename Fp>
    constexpr bool is_approximately_eq(Fp a, Fp b, Fp tolerance = std::numeric_limits<Fp>::epsilon())
    {
        static_assert(std::is_floating_point_v<Fp>, "only floating-point types are supported");
        
        Fp diff = std::abs(a - b);
        return (diff <= tolerance) || 
               (diff < std::max(std::abs(a), std::abs(b)) * tolerance); 
    }

	template <typename Dest, typename Source>
	constexpr Dest round_if(Source value, bool condition) noexcept
	{
		return static_cast<Dest>(condition ? std::round(value) : value);
	}

    template <typename Iter, typename IterCategory>
    POLLUX_MATH_CONCEPT is_iterator_category = is_same_v<typename std::iterator_traits<Iter>::iterator_category, IterCategory>;

    template <typename Iter>
    POLLUX_MATH_CONCEPT is_random_access_iterator_v = is_iterator_category<Iter, std::random_access_iterator_tag>;

    template <typename Iter>
    POLLUX_MATH_CONCEPT is_forward_iterator_v = is_iterator_category<Iter, std::forward_iterator_tag>;
}

namespace pollux::math
{
	template <std::size_t N, typename T>
	struct vec;

    inline namespace defs
    {
        using vec2i   = vec<2, int32_t>;
        using vec2u   = vec<2, uint32_t>;
        using vec2i64 = vec<2, int64_t>;
        using vec2u64 = vec<2, uint64_t>;
        using vec2f   = vec<2, float>;
        using vec2d   = vec<2, double>;

        using vec3i   = vec<3, int32_t>;
        using vec3u   = vec<3, uint32_t>;
        using vec3i64 = vec<3, int64_t>;
        using vec3u64 = vec<3, uint64_t>;
        using vec3f   = vec<3, float>;
        using vec3d   = vec<3, double>;

        using vec4i   = vec<4, int32_t>;
        using vec4u   = vec<4, uint32_t>;
        using vec4i64 = vec<4, int64_t>;
        using vec4u64 = vec<4, uint64_t>;
        using vec4f   = vec<4, float>;
        using vec4d   = vec<4, double>;
    }

    POLLUX_MATH_CONSTINIT auto only_ret = [](const auto& value) { return value; };

    [[nodiscard]]
    POLLUX_MATH_CONSTEXPR float inv_sqrt(float number) noexcept
    {
        using u32 = uint32_t;
        using f32 = float;
        using detail::bit_cast;

        static_assert(sizeof(f32) == sizeof(u32));
        static_assert(std::numeric_limits<f32>::is_iec559, "float type requires iec559");

        u32 i = bit_cast<u32>(number) >> u32{1};
        u32 ii = u32{0x5E5FB414} - i;
        i = u32{0x5F5FB414} - i;

        f32 y  = bit_cast<f32>(i);
        f32 yy = bit_cast<f32>(std::move(ii));
        y = yy * (4.76410007f - number * y * y);
        
        f32 c{ number * y };
        f32 r{ std::fmaf(y, c, -1.0f) };

        c =  std::fmaf(0.374000013f, r, -0.5f);
        return std::fmaf(r * y, c, y);
    }

    [[nodiscard]] 
    POLLUX_MATH_CONSTEXPR double inv_sqrt(double number) noexcept
    {
        using u64 = uint64_t;
        using f64 = double;
        using detail::bit_cast;

        static_assert(sizeof(u64) == sizeof(f64));

        u64 i = bit_cast<u64>(number);
        u64 ix = i - 0x8010000000000000;
        i >>= 1ULL;

        u64 ii = 0x5FCBF6D9DB9A45CD - i;
        i = 0x5FEBF6D9DB9A45CD - i;

        f64 y  = bit_cast<f64>(i);
        y = bit_cast<f64>(std::move(ii)) * (4.7642670025852993 - number * y * y);

        f64 t = std::fma(bit_cast<f64>(std::move(ix)), 
                         y * y, 0.50000031697852854);

        y = std::fma(y, std::move(t), y);

        f64 r = std::fma(y, number * y, -1.0);
        y = std::fma(r * y, std::fma(0.375, std::move(r), -0.5), y);

        return y;
    }

    template <typename T = void>
    struct square_root_functor
    {
        [[nodiscard]]
        constexpr T operator()(T value) const noexcept 
        { 
            static_assert(std::is_floating_point_v<T>, "The type needs to be a floating point type");
            return std::sqrt(value); 
        }
    };

    template<>
    struct square_root_functor<void>
    {
        template <typename T>
        [[nodiscard]]
        constexpr T operator()(T value) const noexcept 
        { 
            static_assert(std::is_floating_point_v<T>, "The type needs to be a floating point type");
            return std::sqrt(value); 
        }
    };

    template <typename T = void>
    struct inverse_square_root_functor
    {
        [[nodiscard]]
        constexpr T operator()(T value) const noexcept 
        { 
            static_assert(std::is_floating_point_v<T>, "The type needs to be a floating point type");
            return inv_sqrt(value); 
        }
    };

    template<>
    struct inverse_square_root_functor<void>
    {
        template <typename T>
        [[nodiscard]]
        constexpr T operator()(T value) const noexcept 
        { 
            static_assert(std::is_floating_point_v<T>, "The type needs to be a floating point type");
            return inv_sqrt(value); 
        }
    };

    template <std::size_t N, typename T>
    struct vec
    {
        using this_type         = vec;
        using value_type        = detail::remove_cvref_t<T>;
        using size_type         = std::size_t;
        using reference         = value_type&;
        using const_reference   = const value_type&;
        using pointer           = value_type*;
        using const_pointer     = const value_type*;
        using iterator          = typename std::array<T, N>::iterator;

        std::array<T, N> components;

        constexpr vec(const vec&) = default;

    #ifdef __cpp_lib_concepts
        template <std::forward_iterator Iter> 
    #else
        template <typename Iter, 
                  typename = std::enable_if_t<detail::is_forward_iterator_v<Iter>>>
    #endif
        constexpr vec(Iter first, Iter last)
        {
			using iter_type = typename std::iterator_traits<detail::remove_cvref_t<Iter>>::value_type;
			using diff_type = typename std::iterator_traits<detail::remove_cvref_t<Iter>>::difference_type;

			static_assert(detail::is_same_v<iter_type, value_type>,
						  "iterator value type mismatch with vector type");
			
			assert(std::distance(first, last) == diff_type(N));
			std::copy(first, last, begin());
        }

    #ifdef __cpp_lib_concepts
        template <template<typename, size_t> 
                  typename ArrayType,
                  typename U, size_t M>
        requires (detail::is_same_v<T, U> && N == M)
    #else
        template <template<typename, size_t> 
                  typename ArrayType,
                  typename U, size_t M,
                  typename = std::enable_if_t<detail::is_same_v<T, U> && N == M>>
    #endif
        explicit constexpr vec(const ArrayType<U, M>& arr)
            : vec(std::begin(arr), std::end(arr))
        {
        }

    #ifdef __cpp_lib_concepts
        template <typename U, size_t M>
        requires (detail::is_same_v<T, U> && N == M)
    #else
        template <typename U, size_t M, 
                  typename = std::enable_if_t<detail::is_same_v<T, U> && N == M>>
    #endif
        explicit constexpr vec(const U (&arr)[M])
            : vec(std::begin(arr), std::end(arr))
        {
        }

        /** 
         * TODO(garcia): enable implicit conversion?
         * if yes, then:
         *  detail::all_convertible_v<T, Args...>
         *  static_cast<T>(args)...
         */
    #ifdef __cpp_lib_concepts
        template <typename... Args> 
        requires detail::all_same_v<T, Args...>
    #else
        template <typename... Args, 
                  typename = std::enable_if_t<detail::all_same_v<T, Args...>>> 
    #endif
        constexpr vec(Args... args)
            : components{{ std::forward<T>(args)... }}
        {
        }

        constexpr vec(const vec& u, const vec& v) noexcept
        {
            *this = v - u;
        }

        template <typename SquareRootFn = square_root_functor<>>
        [[nodiscard]]
        constexpr value_type length(SquareRootFn&& sqrt_fn = {}) const noexcept
        {
            using F = std::conditional_t<std::is_floating_point_v<value_type>, 
										 value_type, double>;
            
			value_type result{ (*this) * (*this) };
			F squared = sqrt_fn(static_cast<F>(result));

            return detail::round_if<value_type>(
				squared, !std::is_floating_point_v<value_type>
			);
        }

        [[nodiscard]]
        constexpr auto& cross(const vec& v) noexcept
        {
            static_assert(N == size_type{3},
                          "The cross product is only valid for three-dimensional vectors");

            const auto [u1, u2, u3] = components;
            const auto [v1, v2, v3] = v.components;

            (*this)[0] = (u2 * v3) - (u3 * v2);
            (*this)[1] = (u3 * v1) - (u1 * v3);
            (*this)[2] = (u1 * v2) - (u2 * v1);

            return *this;
        }

        [[nodiscard]]
        constexpr value_type dot(const vec& v) const noexcept
        {
            return std::inner_product(v.cbegin(), v.cend(), v.cbegin(), value_type{0});
        }

        constexpr auto& lerp(const vec& v, const vec& u, value_type t) noexcept
        {
            return *this = v + ((u - v) * t);
        }

        constexpr vec operator- () const noexcept
        {
            vec temp;
            std::transform(cbegin(), cend(), temp.begin(), std::negate<>{});
            return temp;
        }

        constexpr vec& operator+= (const vec& v) & noexcept
        {
            return *this = *this + v;
        }

        constexpr vec& operator+= (value_type value) & noexcept
        {
            return *this = *this + value;
        }

        constexpr vec& operator-= (const vec& v) & noexcept
        {
            return *this = *this - v;
        }

        constexpr vec& operator-= (value_type value) & noexcept
        {
            return *this = *this - value;
        }
        
        constexpr vec& operator*= (value_type n) & noexcept
        {
            return *this = *this * n;
        }

        constexpr vec& operator/= (value_type n) & noexcept
        {
            return *this = *this / n;
        }

        template <typename InverseSquareRootFn = inverse_square_root_functor<>>
        constexpr auto& norm(InverseSquareRootFn&& inv_sqrt_fn = {}) noexcept
        {
            using FP = std::conditional_t<std::is_floating_point_v<value_type>, 
										  value_type, double>;
            
            // ||v||^2 == v * v
            value_type len{ (*this) * (*this) };
            
			if (len > value_type{})
			{
				value_type value = detail::round_if<value_type>(
					inv_sqrt_fn(static_cast<FP>(len)),
					!std::is_floating_point_v<value_type>
				);

				(*this) *= std::move(value);
			}

            return *this;
        }

        [[nodiscard]]
        constexpr value_type dist(const vec& v) const noexcept
        {
            return vec{ *this, v }.length();
        }

        [[nodiscard]]
        constexpr std::size_t size() const noexcept
        {
            return N;
        }

        [[nodiscard]]
        constexpr bool is_orthogonal_with(const vec& vector) const noexcept
        {
            return !(*this * vector);
        }

        constexpr void swap(vec& other) noexcept(noexcept(components.swap(other.components)))
        {
            components.swap(other.components);
        }

        [[nodiscard]]
        constexpr bool is_normalized() const noexcept(noexcept(length()))
        {
            using F = std::conditional_t<std::is_floating_point_v<value_type>, 
										 value_type, double>;

            return detail::is_approximately_eq(static_cast<F>(length()), F{1});
        }

        template <typename InverseSquareRootFn = inverse_square_root_functor<>>
        [[nodiscard]]
        constexpr auto& scale(value_type factor, InverseSquareRootFn&& inv_sqrt_fn = {}) noexcept
        {
            return norm(std::move(inv_sqrt_fn)) *= factor;
        }

        [[nodiscard]]
        constexpr vec clone() const noexcept
        {
            return *this;
        }

        constexpr auto& clear() noexcept
        {
            components.fill(value_type{});
            return *this;
        }

        [[nodiscard]]
        constexpr auto& operator[](const size_type index) const noexcept
        {
            return components[index];
        }

        [[nodiscard]]
        constexpr auto& operator[](const size_type index) noexcept
        {
            return components[index];
        }

        [[nodiscard]]
        constexpr auto begin() noexcept
        {
            return components.begin();
        }

        [[nodiscard]]
        constexpr auto begin() const noexcept
        {
            return components.begin();
        }

        [[nodiscard]]
        constexpr auto end() noexcept
        {
            return components.end();
        }

        [[nodiscard]]
        constexpr auto end() const noexcept
        {
            return components.end();
        }

        [[nodiscard]]
        constexpr auto cbegin() const noexcept
        {
            return components.cbegin();
        }

        [[nodiscard]]
        constexpr auto cend() const noexcept
        {
            return components.cend();
        }

        [[nodiscard]]
        constexpr auto rbegin() noexcept
        {
            return components.rbegin();
        }

        [[nodiscard]]
        constexpr auto rbegin() const noexcept
        {
            return components.rbegin();
        }

        [[nodiscard]]
        constexpr auto rend() noexcept
        {
            return components.rend();
        }

        [[nodiscard]]
        constexpr auto rend() const noexcept
        {
            return components.rend();
        }

        [[nodiscard]]
        constexpr auto crbegin() const noexcept
        {
            return components.crbegin();
        }

        [[nodiscard]]
        constexpr auto crend() const noexcept
        {
            return components.crend();
        }

        friend constexpr std::ostream& operator<< (std::ostream& os, const vec& v)
        {
            os << "vec<" << N << ", " << detail::name_of<T> << ">: (";

			constexpr size_type n{ N - 1 };
            for (size_type i{}; i < n; ++i)
                os << v[i] << ", ";

            os << v[N - 1] << ")\n";
            return os;
        }

        constexpr operator std::array<T, N>() const noexcept 
        { 
            return components; 
        }
    };

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator+ (const vec<N, T>& u, T n) noexcept
    {
        vec temp(u);
        std::transform(u.begin(), u.end(), temp.begin(), 
                      [n = std::move(n)](const auto& elem) { return elem + n; });
        return temp;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator+ (T n, const vec<N, T>& u) noexcept
    {
        return u + n;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator+ (const vec<N, T>& u, const vec<N, T>& v) noexcept
    {
        vec<N, T> temp;
        std::transform(u.cbegin(), u.cend(), v.cbegin(), temp.begin(), std::plus<>{});
        return temp;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator- (const vec<N, T>& u, T n) noexcept
    {
        vec<N, T> temp;
        std::transform(u.cbegin(), u.cend(), temp.begin(), 
                      [n = std::move(n)](const auto& elem) { return elem - n; });
        return temp;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator- (const vec<N, T>& u, const vec<N, T>& v) noexcept
    {
        vec<N, T> temp;
        std::transform(u.cbegin(), u.cend(), v.cbegin(), temp.begin(), std::minus<>{});
        return temp;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator* (const vec<N, T>& u, T n) noexcept
    {
        vec<N, T> temp(u);
        std::transform(u.cbegin(), u.cend(), temp.begin(), 
                       [n = std::move(n)](const auto& elem) { return elem * n; });
        return temp;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator* (T n, const vec<N, T>& u) noexcept
    {
        return u * n;
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator* (const vec<N, T>& u, const vec<N, T>& v) noexcept
    {
        return u.dot(v);
    }

    template <size_t N, typename T>
    [[nodiscard]]
    constexpr auto operator/ (const vec<N, T>& u, T n) noexcept
    {
        vec<N, T> temp(u);
        std::transform(u.cbegin(), u.cend(), temp.begin(), 
                      [n = std::move(n)](const auto& elem) 
                      { return elem / std::max(n, T{1}); });
        return temp;
    }

    template <size_t N, typename T>
    constexpr bool operator== (const vec<N, T>& u, const vec<N, T>& v)
    {
        return u.components == v.components;
    }

    template <size_t N, typename T>
    constexpr bool operator!= (const vec<N, T>& u, const vec<N, T>& v)
    {
        return !(u == v);
    }

    template <size_t N, typename T>
    constexpr void swap(vec<N, T>& first, vec<N, T>& second) noexcept(noexcept(first.swap(second)))
    {
        first.swap(second);
    }
}

#endif
