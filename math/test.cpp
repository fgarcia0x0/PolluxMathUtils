#include <iostream>
#include "vec.hpp"

namespace plx = pollux::math;

int main()
{
	plx::vec2d a{ 100.0, 100.0 };
	plx::vec2d b = (a) + 0.7;
	plx::vec2d c = 0.7 + a;

	std::cout << a << b << c;
}
