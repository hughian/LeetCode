#include<iostream>
#include<string>
#include<cstdio>
#include<cmath>
using namespace std;

long long max_divisor(long long a, long long b)
{
    return(b == 0 ? a : max_divisor(b, a%b));
}

void print(long long a, long long b)
{
	
    if (b == 0)
    {
        cout << "Inf";
        return;
    }
    if (a == 0)
    {
        cout << 0;
        return;
    }
    if (a < 0)
        cout << "(";
    if (b == 1)
    {
        cout << a;
    }
    else 
    {   
        if (abs(a)>b)
        {
            cout << a / b << " " << ((long long)abs(a) % b) << "/" << b;
        }
        else 
        {
            cout << a << "/" << b;
        }

    }

    if (a < 0)
        cout << ")";



}

void func(long long &a, long long &b)
{
    long long max_div = max_divisor(abs(a), abs(b));
    a = a / max_div;
    b = b / max_div;
    if (b < 0)
    {
        a = -a;
        b = -b;
    }

}

int main()
{
    long long a1, b1, a2, b2;
    scanf("%lld/%lld %lld/%lld", &a1, &b1, &a2, &b2);
    string c="+-*/";
    long long a, b;
    for (int i = 0; i < 4; ++i)
    {
        func(a1, b1);
        print(a1, b1);
        cout << " " << c[i] << " ";
        func(a2, b2);
        print(a2, b2);
        cout << " = ";
        if(c[i]=='+')
        {
            a = a1*b2 + a2*b1;
            b = b1*b2;
        }
        if (c[i] == '-')
        {
            a = a1*b2 - a2*b1;
            b = b1*b2;
        }
        if (c[i] == '*')
        {
            a = a1*a2;
            b = b1*b2;
        }
        if (c[i] == '/')
        {
            a = a1*b2;
            b = b1*a2;
        }
        func(a, b);
        print(a, b);
        cout << endl;
    }
    return 0;
}