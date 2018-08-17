#include<iostream>
#include<vector>
#include<map>
using namespace std;


    int thirdMax(vector<int>& nums) {
        map<int,int> nmap;
        for(int i=0;i<nums.size();i++)
            nmap[nums[i]]++;
        if(nmap.size()<3)
            return (--nmap.end())->first;
        else{
            map<int,int>::iterator it=nmap.end();
            it--;it--;it--;
            return it->first;
        }
    }

int quicksort(vector<int> &v, int left, int right){
        if(left < right){
                int key = v[left];
                int low = left;
                int high = right;
                while(low < high){
                        while(low < high && v[high] > key){
                                high--;
                        }
                        v[low] = v[high];
                        while(low < high && v[low] < key){
                                low++;
                        }
                        v[high] = v[low];
                }
                v[low] = key;
                quicksort(v,left,low-1);
                quicksort(v,low+1,right);
        }
}
void print(vector<int> v){
	for(int i=0;i<v.size();i++)
		cout<<v[i]<<" ";
	cout<<endl;
}
int main()
{
	vector<int> v;
	v.push_back(5);
	v.push_back(2);
	v.push_back(2);
	print(v);
	cout<<thirdMax(v);
	return 0;
}