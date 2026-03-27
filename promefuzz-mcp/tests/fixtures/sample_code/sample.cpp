#include "sample.h"

int test_function(int arg1, char* arg2) {
    return arg1;
}

void another_function() {
    // Implementation
}

int overloaded_func(int x) {
    return x;
}

int overloaded_func(double x) {
    return (int)x;
}

const char* get_name() {
    return "Test";
}

void TestClass::public_method() {
    // Public implementation
}

void TestClass::private_method() {
    // Private implementation
}

void TestClass::protected_method() {
    // Protected implementation
}

int main() {
    test_function(1, "test");
    another_function();
    get_name();
    return 0;
}
