/*
��������
�����ں����Ϸ���n�����ڵľ��Σ�ÿ�����εĿ����1������i��1 �� i �� n�������εĸ߶���hi��
	��n�����ι�����һ��ֱ��ͼ�����磬�������εĸ߶Ⱦͷֱ���3, 1, 6, 5, 2, 3��
�������ҳ��ܷ��ڸ���ֱ��ͼ��������ľ��Σ����ı�Ҫ��������ƽ�С�
    ����������������ӣ������������10��

�����ʽ
������һ�а���һ������n�������ε�����(1 �� n �� 1000)��
�����ڶ��а���n ������h1, h2, �� , hn�����ڵ���֮���ɿո�ָ���(1 �� hi �� 10000)��hi�ǵ�i�����εĸ߶ȡ�
�����ʽ
�������һ�У�����һ��������������ֱ��ͼ�ڵ������ε������

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
            s.push(rect);  //����ʹ�õ�ǰ����߶�����ľ������
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
