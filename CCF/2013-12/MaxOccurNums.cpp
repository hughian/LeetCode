/*
问题描述
　　给定n个正整数，找出它们中出现次数最多的数。
    如果这样的数有多个，请输出其中最小的一个。
输入格式
　　输入的第一行只有一个正整数n(1 ≤ n ≤ 1000)，表示数字的个数。
　　输入的第二行有n个整数s1, s2, …, sn (1 ≤ si ≤ 10000, 1 ≤ i ≤ n)。
    相邻的数用空格分隔。
输出格式
　　输出这n个次数中出现次数最多的数。如果这样的数有多个，输出其中最小的一个。
 */
#include<iostream>
#include<map>
using namespace std;
#define N 10001
class Solution{
public:
    void MaxOccurNums(void){
        int A[N];
        for(int i=0;i<N;i++)
            A[i]=0;
        int n,tmp;
        cin>>n;
        for(int i=0;i<n;i++){
            cin>>tmp;
            A[tmp]++;
        }
        int maxCnt = 0,maxVal =0;
        for(int i=0;i<N;i++){
            if(maxCnt < A[i]){
               maxCnt = A[i];
               maxVal = i;
            }
        }
        cout<<maxVal<<endl;
    }
	void MaxOccurNums_Overload(void){
		map<int,int> mp;
        int n,tmp;
        cin>>n;
        for(int i=0;i<n;i++){
           cin>>tmp;
           mp[tmp]++;
		}
    int maxcnt = 0,maxval;
    for(map<int,int>::iterator it=mp.begin();it!=mp.end();it++){
        if(maxcnt < it->second){
            maxcnt = it->second;
            maxval = it->first;
        }
    }
    cout<<maxval<<endl;
	}
};

int main(void){
    Solution a;
    //solution uses Array
	a.MaxOccurNums();
    //another solution uses map
	a.MaxOccurNums_Overload();
    return 0;
}
