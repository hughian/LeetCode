#include<iostream>
#include<vector>
#include<deque>
#include<algorithm>
#include<string>
using namespace std;

struct Person{
    string name;
    int height;
    Person(string s,int h)
        :name(s),height(h){}
    Person():name(""),height(0){};
};

bool mycmp(Person a,Person b){
    if(a.height == b.height)
        return a.name < b.name;
    else
        return a.height > b.height;
}

int main()
{
    int N,K;
    cin>>N>>K;
    vector<Person> line;
    vector< deque<Person> > group(K);
    string str;
    int height;
    for(int i=0;i<N;i++){
        cin>>str>>height;
        line.push_back(Person(str,height));
    }
    sort(line.begin(),line.end(),mycmp);

    int num = (int)(N/K + 0.5);
    int final_num = 0;
    if(num * K == N)
        final_num = num;
    else if(num * K < N)
        final_num = num + N % K;
    else
        final_num = N % K;
    bool right = true;
    int index = 0;
    for(int i=0;i<final_num;i++){
        if(right)
            group[K-1].push_back(line[index++]);
        else
            group[K-1].push_front(line[index++]);
        right = !right;
    }
    for(int i=K-2;i>=0;i--){
        right = true;
        for(int j=0;j<num;j++){
            if(right)
                group[i].push_back(line[index++]);
            else
                group[i].push_front(line[index++]);
            right = !right;
        }
    }
    
    for(int i=K-1;i>=0;i--){
        for(int j=0;j<(int)group[i].size();j++){
            cout<<group[i][j].name;
            if(j<(int)group[i].size()-1)
                cout<<" ";
        }
        cout<<endl;
    }
    return 0;
}
