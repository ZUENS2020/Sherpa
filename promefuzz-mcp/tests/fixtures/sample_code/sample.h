#ifndef SAMPLE_H
#define SAMPLE_H

int test_function(int arg1, char* arg2);
void another_function();
int overloaded_func(int x);
int overloaded_func(double x);
const char* get_name();

class TestClass {
public:
    void public_method();
private:
    void private_method();
protected:
    void protected_method();
};

#endif
