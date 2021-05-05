# PolluxMathUtils
A utility mathematical library made in modern C ++

## Compiler Requirements
- support at least C++ 17

## How to compile

Every module is a header-only file, so just include the file in your project.
## Example: 
#### **File: test.cpp**
```cpp
#include <pollux/math/vec.hpp>
using namespace pollux::math;

int main()
{
    vec3f u{ 3.f, 2.f, 1.f };
    std::cout << u.length() << '\n';
}
```

**Bulding Example**:

```sh
g++ -Wall -Wpedantic -O2 -std=c++17 test.cpp -o test.exe
```
