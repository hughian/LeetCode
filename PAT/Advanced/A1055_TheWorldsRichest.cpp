#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
int printf(const char *,...);
struct Person{
    string name;
    int age,worth;
    Person():name(""),age(0),worth(0){}
};

bool cmp(Person &a,Person &b){
    if(a.worth == b.worth){
        if(a.age == b.age)
            return a.name < b.name;
        return a.age < b.age;
    }
    return a.worth > b.worth;
}

vector<Person> vec(100010);
int N,K,M;

int main(){
    std::ios::sync_with_stdio(false);//关闭cin与stdin的同步
    cin>>N>>K;
    Person p;
    for(int i=0;i<N;i++){
        cin>>vec[i].name>>vec[i].age>>vec[i].worth;
    }
    sort(vec.begin(),vec.begin()+N,cmp);
    int min,max;
    for(int i=1;i<=K;i++){
        cin>>M>>min>>max;
		//scanf("%d%d%d",&M,&min,&max);
		int cnt = 0;
		printf("Case #%d:\n",i);
		for(int j=0;j<N && cnt<M;j++){
			if(vec[j].age>=min && vec[j].age<=max){
				printf("%s %d %d\n",vec[j].name.c_str(),vec[j].age,vec[j].worth);
				cnt++;
			}
		}
		if(cnt==0)
			printf("None\n");
    }
}
