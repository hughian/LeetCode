
#ifndef _LEETCODE_H_
#define _LEETCODE_H_

#include<iostream>
#include<vector>
#include<stack>
#include<algorithm>
#include<set>
#include<list>
#include<queue>

using namespace std;

void print(vector<int> v){
	for(int i=0;i<v.size();i++)
		cout<<v[i]<<" ";
	cout<<endl;
}
void print2D(vector< vector<int> >& v)
{
	for(int i=0;i<v.size();i++)
		print(v[i]);
}
#endif