/*
问题描述
　　在横轴上放了n个相邻的矩形，每个矩形的宽度是1，而第i（1 ≤ i ≤ n）个矩形的高度是hi。
	这n个矩形构成了一个直方图。例如，六个矩形的高度就分别是3, 1, 6, 5, 2, 3。
　　请找出能放在给定直方图里面积最大的矩形，它的边要与坐标轴平行。
    对于上面给出的例子，最大矩形面积是10。

输入格式
　　第一行包含一个整数n，即矩形的数量(1 ≤ n ≤ 1000)。
　　第二行包含n 个整数h1, h2, … , hn，相邻的数之间由空格分隔。(1 ≤ hi ≤ 10000)。hi是第i个矩形的高度。
输出格式
　　输出一行，包含一个整数，即给定直方图内的最大矩形的面积。

*/

#include<iostream>
#include<stack>

using namespace std;

typedef struct {
    int w,h;
    void set(int hi,int wi = 1){
        w = wi;
        h = hi;
    }
}mat;

int main()
{
    int maxm=0,mtmp,width;
    int n,ht;
    mat rect;
    stack<mat> s;
    cin>>n;
    cin>>ht;
    rect.set(ht);
    s.push(rect);

    for(int i=0;i<n-1;i++){
        cin>>ht;
        rect.set(ht);
        width = mtmp = 0;
        if(rect.h >= s.top().h)
            s.push(rect);
        else{
            while( !s.empty() && (rect.h < s.top().h)){
                width += s.top().w;
                maxm = ((mtmp = width * s.top().h) > maxm)?mtmp:maxm;
                s.pop();
            }
            width += rect.w;
            rect.w = width;  
            s.push(rect);  //保存使用当前输入高度算出的矩形面积
        }
    }
    width = mtmp = 0;
    while( !s.empty()){
        width += s.top().w;
        if((mtmp=width * s.top().h)>maxm)
            maxm = mtmp;
        s.pop();
    }
    cout<<maxm<<endl;
    return 0;
}
