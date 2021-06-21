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
#include <string>
#include <cstring>
#include <algorithm>
#include <functional>

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
    constexpr inline bool is_any_v = std::disjunction_v<std::is_same<T, U>...>;

    template <typename T, typename... Rest>
    constexpr inline bool all_same_v = std::conjunction_v<std::is_same<T, Rest>...>;

    template <typename T, typename U>
    constexpr inline bool is_same_v = std::is_same_v<T, U>;

    template <typename T, typename... Rest>
    constexpr inline bool all_convertible_v = std::conjunction_v<std::is_convertible<T, Rest>...>;

	template <typename Iter>
	using IterCategory = typename std::iterator_traits<Iter>::iterator_category;

	template <typename Iter, typename Category = std::input_iterator_tag>
	constexpr inline bool is_iterator_v = std::is_convertible_v<IterCategory<std::decay_t<Iter>>, Category>;

    /**
     * @brief      Verifies that two floating point values are equals with a
     *             certain tolerance (rounding error).
	 * 
	 * @pre The value of a and b are not NAN'S (Not a Number)
	 * @see https://en.wikipedia.org/wiki/NaN
     *
     * @param[in]  a          The first value
     * @param[in]  b          The second value
     * @param[in]  tolerance  The rounding error tolerance
     *
     * @tparam     Fp         Any float-point type such as float, double or long double
     * @see        https://embeddeduse.com/2019/08/26/qt-compare-two-floats/
     *
     * @return     True if equals, false otherwise.
     */
    template <typename Fp>
    constexpr bool is_approximately_eq(Fp a, Fp b, Fp tolerance = std::numeric_limits<Fp>::epsilon())
    {
        static_assert(std::is_floating_point_v<Fp>, 
					  "only floating-point types are supported");

        /**
		 * 							INFO
         * If two numbers are very close and, hence, the absolute value of their 
         * difference is smaller than epsilon, then these two numbers should be 
         * considered equal.
		 * 
         */
        const Fp diff = std::abs(a - b);
        if (diff <= tolerance)
            return true;

        return diff <= tolerance * std::max(std::abs(a), std::abs(b));
    }

	template <typename Dest, typename Source>
	constexpr Dest round_if(Source value, bool condition) noexcept
	{
		return static_cast<Dest>(condition ? std::round(value) : value);
	}
}

namespace pollux::math
{
	// Forward Declarator
	template <std::size_t N, typename T>
	struct vec;

	// vector typedef's namespace
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

	// lambda functor which only returns the passed value
    static constexpr auto only_ret = [](const auto& value) { return value; };

	/**
	 * @brief Wrapper functor for square root
	 */
	struct sqrt_functor
	{
		template <typename T>
		constexpr T operator()(T value) const noexcept
		{
			static_assert(std::is_floating_point_v<T>, 
					  "The type needs to be a floating point type");

			return std::sqrt(value);
		}
	};

	/**
	 * @brief Wrapper functor for inverse square root (eg: 1 / sqrt(x))
	 */
	struct rsqrt_functor
	{
		template <typename T>
		constexpr T operator()(T value) const noexcept
		{
			static_assert(std::is_floating_point_v<T>,
					  	  "The type needs to be a floating point type");

			return static_cast<T>(1.0) / std::sqrt(value);
		}
	};

	/**
	 * @brief Wrapper functor clamp function
	 */
    struct clamp_functor
    {
        template <typename T>
        constexpr const T& operator()(const T& value, const T& min, const T& max) const noexcept
        {
            return std::clamp(value, min, max);
        }
    };

    /**
     * @brief      An N-dimensional math vector container
     *
     * @tparam     N     represents the number of vector dimensions
     * @tparam     T     represents the type of vector components
     */
    template <std::size_t N, typename T>
    struct vec
    {
		/* Type Assertions */
		static_assert(!std::is_reference_v<T>, "invalid type for vector element");
		static_assert(!std::is_pointer_v<T>,   "invalid type for vector element");

		/*------------------- Type Definitions -----------------------*/
        using this_type         = vec;
        using value_type        = std::decay_t<T>;
        using size_type         = std::size_t;
        using reference         = value_type&;
        using const_reference   = const value_type&;
        using pointer           = value_type*;
        using const_pointer     = const value_type*;
        using iterator          = typename std::array<T, N>::iterator;
		/*------------------------------------------------------------*/

		/**** Vector Components *****/
        std::array<T, N> components;

        /**
         * @brief Adquire the normal vector
         * @param identity_value the identity value (by default is one)
         * @return The normal vector
         */
		[[nodiscard]]
        static constexpr vec normal(const value_type& identity_value = value_type(1)) noexcept
        {
        	vec temp{ };
        	temp[N - 1] = identity_value;
        	return temp;
        }

		/* Default Constructor */
        constexpr vec() = default;

		/* Default Constructor Assignment */
        constexpr vec(const vec&) = default;
		constexpr vec(vec&&) = default;

		/* Default Assignment Operator */
		vec& operator= (const vec&) = default;
		vec& operator= (vec&&) = default;

		/**
		 * @brief Construct vector from wrapper array type
		 * @tparam U The type of array
		 * @tparam M The size of array
		 * @param arr The input array
		 */
		template <template<typename, size_t> typename ArrayType, typename U, size_t M,
				  typename = std::enable_if_t<detail::is_same_v<T, U> && N == M>>
        explicit constexpr vec(const ArrayType<U, M>& arr)
            : vec(std::begin(arr), std::end(arr))
        {
        }

		/**
		 * @brief Construct vector from raw array type
		 * @tparam U The type of array
		 * @tparam M The size of array
		 * @param arr The input array
		 */
		template <typename U, size_t M,
				  typename = std::enable_if_t<detail::is_same_v<T, U> && N == M>>
        explicit constexpr vec(const U (&arr)[M])
            : vec(std::begin(arr), std::end(arr))
        {
        }

		/**
		 * @brief Construct vector from list of elements of type T and size N
		 */
		template <typename... Args,
				  typename = std::enable_if_t<detail::all_convertible_v<T, Args...>>>
        explicit constexpr vec(Args&&... args) noexcept
            : components{{ static_cast<T>(args)... }}
        {
        }

		/**
		 * @brief Construct vector follow the formula: w = (v - u)
		 */
        constexpr vec(const vec& u, const vec& v) noexcept
        {
            *this = v - u;
        }

        /**
         * @brief      Construct vector elements from the [first, last) in such
         *             a way that the length of the interval is equal's to N
         *
         * @param      first  The begin of range of elements
         * @param      last   The end of range of elements
         *
         * @tparam     Iter   The iterator type
         */
        
        template <typename Iter,
                  typename = std::enable_if_t<!detail::is_same_v<Iter, value_type>>,
				  typename = std::enable_if_t<detail::is_iterator_v<Iter, std::forward_iterator_tag>>>
        constexpr vec(Iter first, Iter last) noexcept
        {
            using iter_type = typename std::iterator_traits<std::decay_t<Iter>>::value_type;
            using diff_type = typename std::iterator_traits<std::decay_t<Iter>>::difference_type;

            static_assert(detail::is_same_v<iter_type, value_type>,
                          "iterator value type mismatch with vector type");
            
            assert(std::distance(first, last) == diff_type(N));
            std::copy(first, last, begin());
        }
        

		/**
         * @brief      Calculate the length (or magnitude) of this vector
         *
         * @param      sqrt_fn  Optionally customized square root
         *                      function/functor
         *
         * @tparam     SqrtFn   The type of function/functor implementation of
         *                      square root
         *
         * @return     The length of vector
         */
		template <typename SqrtFn = sqrt_functor>
        [[nodiscard]]
        constexpr value_type length(SqrtFn&& sqrt_fn = {}) const noexcept
        {
            using F = std::conditional_t<std::is_floating_point_v<value_type>, 
										 value_type, double>;
            
			value_type result = this->dot(*this);
			F squared = std::forward<SqrtFn>(sqrt_fn)(static_cast<F>(result));

            return detail::round_if<value_type>(
				std::move(squared), 
				!std::is_floating_point_v<value_type>
			);
        }

        /**
         * @brief      Perform the cross product
         *
         * @param[in]  v The other vector
         *
         * @return     A new vector of cross product vector with v
         */
        [[nodiscard]]
        constexpr vec& cross(const vec& v) noexcept
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

        /**
         * @brief      Calculate the dot product
         *
         * @param[in]  v The other vector
         *
         * @return     The result value of dot product 
         */
        [[nodiscard]]
        constexpr value_type dot(const vec& v) const noexcept
        {
            return std::inner_product(cbegin(), cend(), v.cbegin(), value_type{0});
        }

        /**
         * @brief      Compute v + t(u - v) i.e. the linear interpolation
         *             between v and u for the parameter t
         *
         * @param[in]  v     The first vector
         * @param[in]  u     The second vector
         * @param[in]  t     The tolerance value
         *
         * @return     The vector wich result of linear interpolation
         */
        constexpr vec& lerp(const vec& v, const vec& u, value_type t) noexcept
        {
            return *this = v + (t * (u - v));
        }

        /***** Vector Operators *****/

        constexpr vec operator- () const noexcept
        {
            vec temp;
            std::transform(cbegin(), cend(), temp.begin(), std::negate<>{});
            return temp;
        }

        constexpr vec& operator+= (const vec& v) & noexcept
        {
            return *this = std::move(*this) + v;
        }

        constexpr vec& operator+= (value_type value) & noexcept
        {
            return *this = std::move(*this) + value;
        }

        constexpr vec& operator-= (const vec& v) & noexcept
        {
            return *this = std::move(*this) - v;
        }

        constexpr vec& operator-= (value_type value) & noexcept
        {
            return *this = std::move(*this) - value;
        }
        
        constexpr vec& operator*= (value_type n) & noexcept
        {
            return *this = std::move(*this) * n;
        }

        constexpr vec& operator/= (value_type n) & noexcept
        {
            return *this = std::move(*this) / n;
        }

        /*********************************************/

        /**
         * @brief      Perform vector normalization.
         *
         * @param      rsqrt_fn  Optionally customized inverse square root function/functor
         *
         * @tparam     RSqrtFn   The type of function/functor implementation of inverse square root
         *
         * @return     A reference for this instance
         */
		template <typename RSqrtFn = rsqrt_functor>
        [[maybe_unused]]
        constexpr vec& norm(RSqrtFn&& rsqrt_fn = {}) noexcept
        {
            using FP = std::conditional_t<std::is_floating_point_v<value_type>, 
										  value_type, double>;
            
            // ||v||^2 == v * v
            value_type len{ this->dot(*this) };
            
			if (len > value_type{})
			{
				FP inv_len = std::forward<RSqrtFn>(rsqrt_fn)(static_cast<FP>(len));
				constexpr bool cond = !std::is_floating_point_v<value_type>;

				for (value_type& component : (*this))
					component = detail::round_if<value_type>(component * inv_len, cond);
			}

            return *this;
        }

        /**
         * @brief      Calculate the distance of this vector in relation to the other
         *
         * @param[in]  v  The other vector
         *
         * @return     The value of distance
         */
        [[nodiscard]]
        constexpr value_type dist(const vec& v) const noexcept
        {
            return vec{ *this, v }.length();
        }

        /**
         * @brief      Return the number of dimensions of vector
         *
         * @return     The number of dimensions of vector
         */
        [[nodiscard]]
        constexpr std::size_t size() const noexcept
        {
            return N;
        }

        /**
         * @brief      Determines whether the specified vector is orthogonal
         *             with other vector.
         *
         * @param[in]  other  The other vector
         *
         * @return     True if the specified vector is orthogonal with other,
         *             False otherwise.
         */
        [[nodiscard]]
        constexpr bool is_orthogonal_with(const vec& other) const noexcept
        {
            return !(*this * other);
        }

        /**
         * @brief      Swap this vector with another
         *
         * @param      other  The other vector
         */
        constexpr void swap(vec& other) noexcept(noexcept(components.swap(other.components)))
        {
            components.swap(other.components);
        }

        /**
         * @brief      Verify if vector is normalized
         *
         * @return     True if normalized, False otherwise.
         */
        [[nodiscard]]
        constexpr bool is_normalized() const noexcept(noexcept(length()))
        {
            using F = std::conditional_t<std::is_floating_point_v<value_type>, 
										 value_type, double>;

            return detail::is_approximately_eq(static_cast<F>(length()), F{1});
        }

        /**
         * @brief      Scale this vector from a scalar value called factor
         *
         * @param[in]  factor    The factor to be scaled
         * @param      rsqrt_fn  Optionally customized inverse square root
         *                       function/functor
         *
         * @tparam     RSqrtFn   The type of function/functor implementation of
         *                       inverse square root
         *
         * @return     A reference for this instance
         */
		template <typename RSqrtFn = rsqrt_functor>
        [[nodiscard]]
        constexpr vec& scale(value_type factor, RSqrtFn&& rsqrt_fn = {}) noexcept
        {
            return norm(std::move(rsqrt_fn)) *= factor;
        }


        /**
         * @brief      Creates a new instance of the object with same properties
         *             than original.
         *
         * @return     Copy of this object.
         */
        [[nodiscard]]
        constexpr vec clone() const noexcept
        {
            return *this;
        }

        /**
         * @brief      Clear vector with the given value.
         *
         * @param[in]  value  The value
         *
         * @return     A reference for this instance
         */
		constexpr vec& clear(const value_type& value = value_type(0)) noexcept
        {
			std::fill(begin(), end(), value);
            return *this;
        }

        /**
         * @brief      Clamps a value between an upper and lower bound.
         *
         * @param[in]  min       The minimum value
         * @param[in]  max       The maximum value
         * @param      clamp_fn  Optionally customized clamp function/functor
         *
         * @tparam     ClampFn   The type of function/functor implementation of
         *                       clamp
         *
         * @return     A reference for this instance
         */
		template <typename ClampFn = clamp_functor>
        constexpr vec& clamp(const value_type& min, const value_type& max,
                             ClampFn&& clamp_fn = {}) noexcept
        {
			for (value_type& value : (*this))
				value = std::forward<ClampFn>(clamp_fn)(std::move(value), min, max);

            return *this;
        }

        /**
         * @brief      Clamps a value between an upper and lower bound.
         *
         * @param[in]  range     The range vector represent [min, max]
         * @param      clamp_fn  Optionally customized clamp function/functor
         *
         * @tparam     ClampFn   The type of function/functor implementation of clamp
         *
         * @return     A reference for this instance
         */
		template <typename ClampFn = clamp_functor>
        constexpr vec& clamp(const vec<2, value_type>& range,
                             ClampFn&& clamp_fn = {}) noexcept
        {
            return clamp(range[0], range[1], std::move(clamp_fn));
        }

        /**
         * @brief      Perform fused multiply–add operation
         *
         * @param[in]  v1    The first vector
         * @param[in]  v2    The second vector
         *
         * @return     A reference for this instance
         */
        constexpr vec& mul_add(const vec& v1, const vec& v2) noexcept
        {
			auto& this_ref = *this;

			for (size_type i = 0; i < N; ++i)
				this_ref[i] = (std::move(this_ref[i]) * v1[i]) + v2[i];

            return this_ref;
        }

        /**
         * @brief      Verifies when two vectors are equal from a certain tolerance
         *
         * @param[in]  v  The vector with who will be compared
         *
         * @return     True if is equivalent, False otherwise.
         */
        template <typename Fp = std::conditional_t<std::is_floating_point_v<value_type>, 
												   value_type, double>>
        [[nodiscard]]
        constexpr bool is_equivalent(const vec& v, 
									 Fp tolerance = std::numeric_limits<Fp>::epsilon()) const noexcept
        {
            using namespace detail;

            bool eq = false;
            for (size_type i = 0; i < N; ++i)
                eq |= is_approximately_eq(Fp((*this)[i]), Fp(v[i]));

            return eq;
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

        /**
         * @brief      Returns an immutable iterator pointing to the first
         *             element in the vector.
         *
         * @return     Iterator to beginning
         */
        [[nodiscard]]
        constexpr auto begin() const noexcept
        {
            return components.begin();
        }

        /**
         * @brief      Returns an iterator pointing to the last element in the vector.
         *
         * @return     Iterator to ending
         *
         */
        [[nodiscard]]
        constexpr auto end() noexcept
        {
            return components.end();
        }

        /**
         * @brief      Returns an immutable iterator pointing to the last
         *             element in the vector.
         *
         * @return     Iterator to ending
         */
        [[nodiscard]]
        constexpr auto end() const noexcept
        {
            return components.end();
        }

        /**
         * @brief      Returns an const iterator pointing to the first element
         *             in the vector.
         *
         * @return     Iterator to beginning
         */
        [[nodiscard]]
        constexpr auto cbegin() const noexcept
        {
            return components.cbegin();
        }

        /**
         * @brief      Returns an const iterator pointing to the last element
         *             in the vector.
         *
         * @return     Iterator to ending
         */
        [[nodiscard]]
        constexpr auto cend() const noexcept
        {
            return components.cend();
        }

        /**
         * @brief      Returns a reverse iterator pointing to the last element
         *             in this vector.
         *
         * @return     reverse iterator to reverse beginning
         */
        [[nodiscard]]
        constexpr auto rbegin() noexcept
        {
            return components.rbegin();
        }

        /**
         * @brief      Returns a immutable reverse iterator pointing to the last
         *             element in this vector.
         *
         * @return     reverse iterator to reverse beginning
         */
        [[nodiscard]]
        constexpr auto rbegin() const noexcept
        {
            return components.rbegin();
        }

        /**
         * @brief      Returns a reverse iterator pointing to the theoretical
         *             element preceding the first element in this vector
         *
         * @return     reverse iterator to reverse end
         */
        [[nodiscard]]
        constexpr auto rend() noexcept
        {
            return components.rend();
        }

        /**
         * @brief      Returns a immutable reverse iterator pointing to the theoretical
         *             element preceding the first element in this vector
         *
         * @return     reverse iterator to reverse end
         */
        [[nodiscard]]
        constexpr auto rend() const noexcept
        {
            return components.rend();
        }

        /**
         * @brief      Returns a const reverse iterator pointing to the last
         *             element in this vector.
         *
         * @return     reverse iterator to reverse beginning
         */
        [[nodiscard]]
        constexpr auto crbegin() const noexcept
        {
            return components.crbegin();
        }

        /**
         * @brief      Returns a const reverse iterator pointing to the theoretical
         *             element preceding the first element in this vector
         *
         * @return     reverse iterator to reverse end
         */
        [[nodiscard]]
        constexpr auto crend() const noexcept
        {
            return components.crend();
        }

        /**
         * @brief      Returns a string representation of the object.
         *
         * @return     String representation of the object.
         */
		std::string to_str() const
		{
			const char* fmt = "vec<%zu, %s>: (";
			std::size_t needed_size = std::snprintf(nullptr, 0, fmt, N, detail::name_of<T>);

			assert(needed_size > 0);

			std::string buffer(needed_size, '\0');

			[[maybe_unused]]
			std::size_t writted_size = std::snprintf(buffer.data(), 
													 needed_size + 1,
													 fmt, N, detail::name_of<T>);

			assert(writted_size && writted_size <= needed_size);

			auto first = cbegin();
			auto last  = cend() - 1U;

			for (; first != last; ++first)
			{
				buffer += std::to_string(*first);
				buffer.push_back(',');
				buffer.push_back(' ');
			}

			buffer += std::to_string(*last);
			buffer.push_back(')');

			return buffer;
		}

        friend constexpr std::ostream& operator<< (std::ostream& os, const vec& v)
        {
			return (os << v.to_str());
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
    constexpr auto operator/ (const vec<N, T>& u, T n) noexcept
    {
		assert(n != 0);
        return u * (T(1.0) / n);
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
