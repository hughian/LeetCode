#include<iostream>
#include<map>
#include<vector>
#include<algorithm>
using namespace std;
char prior[] = {'A','C','M','E'};
struct Student{
    int id;
    int c,m,e,a;
    int score[4];
    int rank[4];
    Student(int _id,int _c,int _m,int _e)
        :id(_id),c(_c),m(_m),e(_e)
    {
        a =( _c + _m + _e ) / 3;
        for(int i=0;i<4;i++)
            rank[i] = 0;
        score[0] = a;
        score[1] = c;
        score[2] = m;
        score[3] = e;
    }
};
int order;
bool id_cmp(Student s1,Student s2)
{
    return s1.id < s2.id;
}

bool cmp(Student s1,Student s2)
{
    return s1.score[order] >= s2.score[order];
}

int find(int id,vector<Student> &v)
{
    int left = 0,right = v.size()-1;
    int mid;
    while(left<=right){
        mid = (left+right) /2;

        if(v[mid].id==id)
            return mid;
        else if(v[mid].id<id)
            left = mid;
        else
            right = mid;
    }
    return -1;
}
int find(vector<Student> & v,int id)
{
	for(int i=0;i<(int)v.size();i++)
	{
		if(v[i].id==id)
			return i;
	}
	return -1;
}

int main()
{
    int N,M;
    vector<Student> vs;
    cin>>N>>M;
    int id,c,m,e;
    for(int i=0;i<N;i++){
        cin>>id>>c>>m>>e;
        vs.push_back(Student(id,c,m,e));
    }
    for(order=0;order<4;order++){
        sort(vs.begin(),vs.end(),cmp);
        vs[0].rank[order] = 1;
        for(int i=1;i<N;i++){ //成绩与前一名相同时,并列,排名也与前一名相同
            if(vs[i].score[order] == vs[i-1].score[order])
                vs[i].rank[order] = vs[i-1].rank[order];
            else
                vs[i].rank[order] = i+1;
        }
    }
    int index = -1;
    sort(vs.begin(),vs.end(),id_cmp);
    for(int i=0;i<M;i++){
        cin>>id;
        
        index = find(vs,id);
        if(index==-1){
            cout<<"N/A"<<endl;
        }else{
            int min = vs[index].rank[0];
            int tag = 0;
            for(int j=1;j<4;j++){
                if(min > vs[index].rank[j]){
                    min = vs[index].rank[j];
                    tag = j;
                }
            }
            cout<<min<<" "<<prior[tag]<<endl;
        }
    }
    return 0;
}
