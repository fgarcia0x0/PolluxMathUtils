# PolluxMathUtils
A utility mathematical library made in modern C ++

## Compiler Requirements
- support at least C++ 17

## How to compile

Every module is a header-only file, so just include the file in your project.
## Example
#### File: test.cpp
```cpp
#include <iostream>
#include <pollux/math/vec.hpp>

using namespace pollux::math::defs;

int main()
{
    vec<3, float> u{ 3.0f, 2.0f, 1.0f }; // or vec3f
    std::cout << u.length() << '\n';
}
```

**Bulding Example**:

```sh
g++ -Wall -Wpedantic -O2 -std=c++17 test.cpp -o test.exe
```

## Supported Operations
- Sum
- Subtraction
- Multiplication
- Division
- Dot Product
- Cross Product
- length or magnitude
- normalization
- distance
- size
- swap
- orthogonality
- scale
- clone
- clear
- Access individual elements
- linear interpolation (lerp)
- normal vector
- to_str
- is_equivalent
