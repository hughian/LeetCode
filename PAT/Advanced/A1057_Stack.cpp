#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cstdio>
#include<set>
using namespace std;
vector<int> stk;
multiset<int> mins,maxs;
int mid;
int Adjust(){
    multiset<int>::iterator it;
    if(mins.size()>maxs.size()+1){
        it = mins.end();it--;
        maxs.insert(*it);
        mins.erase(it);
    }else if(mins.size()<maxs.size()){
        it = maxs.begin();
        mins.insert(*it);
        maxs.erase(it);
    }
    if(stk.size()>0){
        it = mins.end();it--;
        mid = *it;
    }
}
int main()
{
    int n;
	char buf[100];
    scanf("%d",&n);
    string op;
    int key,t;
    multiset<int>::iterator pos;
    for(int i=0;i<n;i++){ 	
        scanf("%s",buf);
		op = string(buf);
        if(op=="Push"){
            scanf("%d",&key);
            if(stk.size()==0){
                mins.insert(key);
                mid = key;
            }else if(key <= mid)
                mins.insert(key);
            else
                maxs.insert(key);
            stk.push_back(key);
            Adjust();
        }else if(op == "Pop"){
            if(stk.size()==0){
                printf("Invalid\n");
            }else{
                t = stk.back();
                stk.pop_back();
                if(mid>=t){
                    pos = mins.find(t);
                    mins.erase(pos);
                }else{
                    pos = maxs.find(t);
                    maxs.erase(pos);
                }
                Adjust();
                printf("%d\n",t);
            }
        }else{
            if(stk.size()==0)
                printf("Invalid\n");
            else
                printf("%d\n",mid);
        }
    }
    return 0;
}
