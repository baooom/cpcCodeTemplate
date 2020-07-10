

[TOC]

#  代码头

```c++
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <vector>
#include <string>
#include <iostream>
#include <list>
#include <cstdlib>
#include <bitset>
//#define getchar() (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 21, stdin), p1 == p2) ? EOF : *p1++)
//char buf[(1 << 21) + 1], * p1 = buf, * p2 = buf;
//#define int long long
#define lowbit(x) (x&(-x))
#define lson root<<1,l,mid
#define rson root<<1|1,mid+1,r
#define pb push_back
typedef unsigned long long ull;
typedef long long ll;
typedef pair<int, int> pii;
#define bug puts("BUG");
const long long INF = 0x3f3f3f3f3f3f3f3fLL;
const int inf = 0x3f3f3f3f;
const int mod = 1000000007;
const double eps = 1e-6;
template<class T>inline void read(T& x) {
	int sign = 1; char c = getchar(); x = 0;
	while (c > '9' || c < '0') { if (c == '-')sign = -1; c = getchar(); }
	while (c >= '0' && c <= '9') { x = x * 10 + c - '0'; c = getchar(); }
	x *= sign;
}
using namespace std;

```

#  数据结构

## 线段树合并

```c++
int merge(int u, int v, int l, int r)
{
    if(!u)return v;if(!v)return u;
    if (l == r)
    {
        val[u].first += val[v].first;
        val[u].second = l;
        return u;
    }
    int mid = l + r >> 1;
    ls[u] = merge(ls[u], ls[v], l, mid);
    rs[u] = merge(rs[u], rs[v], mid + 1, r);
    pushup(u);
    return u;
}

```

##  可持久化线段树

```c++
struct Seg
{
	int v, ls, rs;
}val[maxn * 20];
int rot[maxn], cnt;
inline void pushup(int root)
{
	val[root].v = val[val[root].ls].v + val[val[root].rs].v;
}
int buildemp(int l, int r)
{
	int id = cnt++;
	if (l == r)
	{
		val[id].v = 0;
		return id;
	}
	int mid = l + r >> 1;
	val[id].ls = buildemp(l, mid);
	val[id].rs = buildemp(mid + 1, r);
	pushup(id);
	return id;
}

int addnode(int pre, int l, int r, int v)
{
	int id = cnt++;
	if (l == r)
	{
		val[id].v = val[pre].v + 1;
		return id;
	}
	int mid = l + r >> 1;
	val[id] = val[pre];
	if (v <= mid)
		val[id].ls = addnode(val[pre].ls, l, mid, v);
	else
		val[id].rs = addnode(val[pre].rs, mid + 1, r, v);
	pushup(id);
	return id;
}

int query(int l, int r, int lt, int rt, int k)
{
	if (l == r)return l;
	int tt = val[val[rt].ls].v - val[val[lt].ls].v;
	int mid = l + r >> 1;
	if (tt >= k)
		return query(l, mid, val[lt].ls, val[rt].ls, k);
	else
		return query(mid + 1, r, val[lt].rs, val[rt].rs, k - tt);
}

```

##  平衡树

### splay

```c++
const int maxn = 1e5 + 10;
int n, m, rt, tot;
int fa[maxn], siz[maxn], ch[maxn][2], val[maxn], cnt[maxn];
inline void pushup(int x)
{
	siz[x] = siz[ch[x][0]] + siz[ch[x][1]] + cnt[x];
}

inline void connect(int son, int father, int d)
{
	ch[father][d] = son;
	fa[son] = father;
}

inline int ws(int x)
{
	return ch[fa[x]][1] == x;
}

inline void rot(int x)
{
	int f = fa[x], ff = fa[f];
	int c = ws(x), cc = ws(f);
	connect(ch[x][c ^ 1], f, c);
	connect(f, x, c ^ 1);
	connect(x, ff, cc);
	pushup(f);
	pushup(x);
}

inline void splay(int v, int goal)
{
	while (fa[v] != goal)
	{
		int f = fa[v], ff = fa[f];
		if (ff != goal)
		{
			if (ws(v) == ws(f))rot(f);
			else rot(v);
		}
		rot(v);
	}
	if (goal == 0)rt = v;
}

inline void insert(int x)
{
	int u = rt, f = 0;
	while (u && val[u] != x)
	{
		f = u;
		u = ch[u][x > val[u]];
	}
	if (u)++cnt[u];
	else
	{
		u = ++tot;
		if (f)ch[f][x > val[f]] = u;
		ch[u][0] = ch[u][1] = 0;
		fa[u] = f, val[u] = x, siz[u] = cnt[u] = 1;
	}
	splay(u, 0);
}

inline void find(int x)
{
	int u = rt;
	if (!u)return;
	while (ch[u][x > val[u]] && val[u] != x)
		u = ch[u][x > val[u]];
	splay(u, 0);
}

inline int pre(int x)
{
	find(x);
	int u = rt;
	if (val[u] < x)return u;
	u = ch[u][0];
	while (ch[u][1])u = ch[u][1];
	return u;
}

inline int nxt(int x)
{
	find(x);
	int u = rt;
	if (val[u] > x)return u;
	u = ch[u][1];
	while (ch[u][0])u = ch[u][0];
	return u;
}

inline void delet(int x)
{
	int last = pre(x), next = nxt(x);
	splay(last, 0), splay(next, last);
	int del = ch[next][0];
	if (cnt[del] > 1)--cnt[del], splay(del, 0);
	else ch[next][0] = 0;
}

inline int kth(int x)
{
	int u = rt;
	if (siz[u] < x)return 0;
	while (1)
	{
		int y = ch[u][0];
		if (x > siz[y] + cnt[u])
		{
			x -= siz[y] + cnt[u];
			u = ch[u][1];
		}
		else
		{
			if (x <= siz[y])
				u = y;
			else
				return val[u];
		}
	}
}

```

### treap

at_rank要+1,get_rank要-1，因为最小值是-inf

```c++
const int maxn = 1e5 + 10;
int tot;
int ch[maxn][2];
int val[maxn], dat[maxn];
int sz[maxn], cnt[maxn];
int root;

int newnode(int v)
{
    val[++tot] = v;
    dat[tot] = rand();
    sz[tot] = 1;
    cnt[tot] = 1;
    return tot;
}

void pushup(int x)
{
    sz[x] = sz[ch[x][0]] + sz[ch[x][1]] + cnt[x];
}

void init()
{
    root = newnode(-inf);
    ch[root][1] = newnode(inf);
    pushup(root);
}

void Rot(int &id, int d)
{
    int temp = ch[id][d ^ 1];
    ch[id][d ^ 1] = ch[temp][d];
    ch[temp][d] = id;
    id = temp;
    pushup(ch[id][d]);
    pushup(id);
}

void insert(int &id,int v)
{
    if (!id)
    {
        id = newnode(v);
        return;
    }
    if(v==val[id])
        cnt[id]++;
    else
    {
        int d = !(v < val[id]);
        insert(ch[id][d], v);
        if (dat[id] < dat[ch[id][d]])
            Rot(id, d ^ 1);
    }
    pushup(id);
}

void del(int &id, int v)
{
    if(!id)
        return;
    if (v == val[id])
    {
        if (cnt[id] > 1)
        {
            --cnt[id];
            pushup(id);
            return;
        }
        if (ch[id][0] || ch[id][1])
        {
            if (!ch[id][1] || dat[ch[id][0]] > dat[ch[id][1]])
            {
                Rot(id, 1);
                del(ch[id][1], v);
            }else
            {
                Rot(id, 0);
                del(ch[id][0], v);
            }
            pushup(id);
        }else
        {
            id = 0;
        }
        return;
    }
    if(v<val[id])
        del(ch[id][0], v);
    else
        del(ch[id][1], v);
    pushup(id);
    return;
}

int get_rank(int id, int v)
{
    if(!id)
        return 0;
    if (v == val[id])
        return sz[ch[id][0]] + 1;
    else if (v < val[id])
        return get_rank(ch[id][0], v);
    else
        return sz[ch[id][0]] + cnt[id] + get_rank(ch[id][1], v);
}

int at_rank(int id, int k)
{
    if(!id)
        return inf;
    if (k <= sz[ch[id][0]])
        return at_rank(ch[id][0], k);
    else if (k <= sz[ch[id][0]] + cnt[id])
        return val[id];
    else
        return at_rank(ch[id][1], k - sz[ch[id][0]] - cnt[id]);
}

int get_next(int v)
{
    int cur = root, res;
    while (cur)
    {
        if (val[cur] > v)
        {
            res = val[cur];
            cur = ch[cur][0];
        }else
        {
            cur = ch[cur][1];
        }
    }
    return res;
}

int get_pre(int v)
{
    int cur = root, res;
    while (cur)
    {
        if (val[cur] < v)
        {
            res = val[cur];
            cur = ch[cur][1];
        }else
        {
            cur = ch[cur][0];
        }
    }
    return res;
}
```



##  树链剖分

$$
直径d=\max\{dis_{u,v},dis_{u,p},dis_{v,p}\}\\
u,v为原直径，p为新增点
$$



```c++
int dep[maxn], sz[maxn], f[maxn], son[maxn], top[maxn], dfn[maxn], idx;
void dfs1(int u,int pre,int d)
{
	f[u] = pre;
	dep[u] = d;
	sz[u] = 1;
	for (int i=head[u];~i;i=edge[i].next)
	{
		int v = edge[i].to;
        if(v==pre)continue;
		dfs1(v, u,d + 1);
        if (sz[v] > sz[son[u]])son[u]=v;
		sz[u] += sz[v];
	}
}
void dfs2(int u, int to)
{
	top[u] = to;
	dfn[u] = ++idx;
	if(son[u])
		dfs2(son[u], to);
	for (int i=head[u];~i;i=edge[i].next)
	{
        int v=edge[i].to;
        if(v==f[u]||v==son[u])
            continue;
		dfs2(v, v);
	}
}

int LCA(int u, int v)
{
	while (top[u] != top[v])
	{
		if (dep[top[u]] < dep[top[v]])swap(u, v);
		u = f[top[u]];
	}
	return dep[u] < dep[v] ? u : v;
}
```

##  虚树

```c++
void insert(int u){
    if(top <= 1) {stk[++top] = u;return ;}
    int lca = LCA(u,stk[top]);
    if(lca == stk[top]) {stk[++top] = u;return ;}
    while(top > 1 && dfn[lca] <= stk[top-1]) {
        addedge(stk[top-1],stk[top]);
        --top;
    }
    if(lca != stk[top]) stk[++top] = lca;
    stk[++top] = u;
}
```

##  树套树(树状数组套权值线段树)

```c++
int k;
const int maxn = 2e5 + 10;
struct NODE
{
    int x, y, z;
    int ansidx;
    bool operator < (const NODE& other)const
    {
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }
    bool operator == (const NODE& other)const
    {
        return x == other.x && y == other.y && z == other.z;
    }

} node[maxn];
int nodez[maxn * 90], tot, ls[maxn * 90], rs[maxn * 90];
int rt[maxn << 2];
int ans[maxn], ansout[maxn];
void pushup(int root)
{
    nodez[root] = nodez[ls[root]] + nodez[rs[root]];
}
void updatez(int root, int l, int r, int z, int v)
{
    if (l == r)
    {
        nodez[root] += v;
        return;
    }
    int mid = l + r >> 1;
    if (z <= mid)
    {
        if(!ls[root])
            ls[root] = ++tot;
        updatez(ls[root], l, mid, z, v);
    }
    else
    {
        if(!rs[root])
            rs[root] = ++tot;
        updatez(rs[root], mid + 1, r, z, v);
    }
    pushup(root);
}

void updatey(int y,int z, int v)
{
    for (; y <= k; y += lowbit(y))
    {
        if(!rt[y])
            rt[y] = ++tot;
        updatez(rt[y], 1, k, z, v);
        pushup(rt[y]);
    }
}

int queryz(int root, int l, int r, int lz, int rz)
{
    if (l == lz && r == rz)
    {
        return nodez[root];
    }
    int mid = l + r >> 1;
    if (rz <= mid)
        return queryz(ls[root], l, mid, lz, rz);
    else if (lz > mid)
        return queryz(rs[root], mid + 1, r, lz, rz);
    else
    {
        return queryz(ls[root], l, mid, lz, mid) + queryz(rs[root], mid + 1, r, mid + 1, rz);
    }
}

int queryy(int y, int lz, int rz)
{
    int res = 0;
    for (; y > 0; y -= lowbit(y))
    {
        if(!rt[y])
            continue;
        res += queryz(rt[y], 1, k, lz, rz);
    }
    return res;
}
```

## 单调栈求最大矩阵

```c++
int ans = 0;
pii v;
for(int i = 1; i <= n; i++){
	for(int j = 1; j <= m; j++){
		if (s.empty() && !a[i][j]) continue;
			
		if (s.empty() || s.top().second <= a[i][j]) s.push(make_pair(j, a[i][j]));
		else {
			while(!s.empty() && s.top().second > a[i][j]){
				v = s.top();
				s.pop();
				int area = (j-v.first)*v.second;
				if (area > ans) ans = area;   
			}
			s.push(make_pair(v.first, a[i][j]));
		}
			
	}
}      
```

## 笛卡尔树

```c++
stk[top = 1] = 0;
for (int i = 1; i <= n; ++i)
{
    read(p[i]);
    while (top && p[i] < p[stk[top]])
        l[i] = stk[top--];
    if (top)
        r[stk[top]] = i;
    stk[++top] = i;
}
//笛卡尔树，小顶堆
```



#  数论

## 扩展欧几里得

$$
ax+by=\gcd(a,b)\\
\frac{ac}{\gcd(a,b)}x+\frac{bc}{\gcd(a,b)}y=c\\
当且仅当\gcd(a,b)|c有解
$$



```c++
ll exgcd(ll a, ll b, ll& x, ll& y) {
	if (b == 0) { x = 1, y = 0; return a; }
	ll r = exgcd(b, a % b, x, y), tmp;
	tmp = x; x = y; y = tmp - (a / b) * y;
	return r;
}
```



##  扩展中国剩余定理

$$
x\equiv c_i(\mod m_i)
$$

```c++
const ll maxn = 1e6 + 10;
ll K, C[maxn], M[maxn], x, y;
ll gcd(ll a, ll b) {
	return b == 0 ? a : gcd(b, a % b);
}
ll exgcd(ll a, ll b, ll& x, ll& y) {
	if (b == 0) { x = 1, y = 0; return a; }
	ll r = exgcd(b, a % b, x, y), tmp;
	tmp = x; x = y; y = tmp - (a / b) * y;
	return r;
}
ll inv(ll a, ll b) {
	ll r = exgcd(a, b, x, y);
	while (x < 0) x += b;
	return x;
}

ll excrt(ll M[], ll C[], ll K)
{
    bool flag = 1;
    for (ll i = 2; i <= K; i++) {
        ll M1 = M[i - 1], M2 = M[i], C2 = C[i], C1 = C[i - 1], T = gcd(M1, M2);
        if ((C2 - C1) % T != 0) { flag = 0; break; }
        M[i] = (M1 * M2) / T;//此处溢出
        C[i] = (inv(M1 / T, M2 / T) * (C2 - C1) / T) % (M2 / T) * M1 + C1;//此处溢出
        C[i] = (C[i] % M[i] + M[i]) % M[i];
    }
    return flag ? C[K] : -1;
}

```

##  线性递推逆元

```c++
int inv[100001];
int p;
int main()
{
	cin >> p;//模数
	inv[1] = 1;//1的逆元为1
	for (int i = 2; i <= 10; i++)
	{
		inv[i] = (p - (p / i)) * inv[p % i] % p;//递推公式
		printf("%d %d %d\n", i, inv[i], (i * inv[i]) % p);
	}
}
```

##  欧拉降幂

$$
\left.\begin{cases}
A^B\equiv A^{B \mod \phi(C)}(\mod C)\qquad B\lt\phi(C)\\
A^B\equiv A^{B \mod \phi(C)+\phi(C)}(\mod C)\qquad B\ge\phi(C)
\end{cases}\right.
$$

```c++
#define MOD(a,b) (a<b?a:a%b+b)
ll phi(ll n)
{
	ll ans = n;
	for (ll i = 2; i * i <= n; ++i)
	{
		if (n % i == 0)
		{
			ans -= ans / i;
			while (n % i == 0)n /= i;
		}
	}
	if (n > 1)ans -= ans / n;
	return ans;
}
ll qmod(ll a, ll n, ll mod)
{
	ll ans = 1;
	while (n)
	{
		if (n & 1)
			ans = MOD(ans * a, mod);
		a = MOD(a * a, mod);
		n >>= 1;
	}
	return ans;
}
ll slove(ll a, ll b, ll mod)
{
	if (b == 0)return MOD(1, mod);
	if (mod == 1)return MOD(a, mod);
	return qmod(a, slove(a, b - 1, phi(mod)), mod);
}
int main()
{
	int t;
	for (read(t); t--;)
	{
		ll a, b, p;
		read(a), read(b), read(p);
		printf("%lld\n", slove(a, b, p) % p);
	}
}
```

##  二次剩余

$$
X^2\equiv d(\mod p)
$$

```c++
#include <time.h>
struct num {
	ll x, y;
};

ll w;

num mul(num a, num b, ll p)
{
	num ans = { 0,0 };
	ans.x = ((a.x * b.x % p + a.y * b.y % p * w % p) % p + p) % p;
	ans.y = ((a.x * b.y % p + a.y * b.x % p) % p + p) % p;
	return ans;
}

ll powwR(ll a, ll b, ll p) {
	ll ans = 1;
	while (b) {
		if (b & 1)ans = 1ll * ans % p * a % p;
		a = a % p * a % p;
		b >>= 1;
	}
	return ans % p;
}
ll powwi(num a, ll b, ll p) {
	num ans = { 1,0 };
	while (b) {
		if (b & 1)ans = mul(ans, a, p);
		a = mul(a, a, p);
		b >>= 1;
	}
	return ans.x % p;
}

ll solve(ll n, ll p)
{
	n %= p;
	if (p == 2)return n;
	if (powwR(n, (p - 1) / 2, p) == p - 1)return -1;//不存在
	ll a;
	while (1)
	{
		a = rand() % p;
		w = ((a * a % p - n) % p + p) % p;
		if (powwR(w, (p - 1) / 2, p) == p - 1)break;
	}
	num x = { a,1 };
	return powwi(x, (p + 1) / 2, p);
}

int main()
{
	srand(time(0));
	int t;
	scanf("%d", &t);
	while (t--)
	{
		ll n, p;
		scanf("%lld%lld", &n, &p);
		if (!n) {
			printf("0\n"); continue;
		}
		ll ans1 = solve(n, p), ans2;
		if (ans1 == -1)printf("Hola!\n");
		else
		{
			ans2 = p - ans1;
			if (ans1 > ans2)swap(ans1, ans2);
			if (ans1 == ans2)printf("%lld\n", ans1);
			else printf("%lld %lld\n", ans1, ans2);
		}
	}
}

```

##  类欧几里得

$$
f(a,b,c,n)=\sum_{i=0}^{i\le n}\left\lfloor\frac{ai+b}{c}\right\rfloor\\
$$

$$
f(a,b,c,n)=\left.\begin{cases}
\frac{a}{c}*\frac{n*(n+1)}{2}+\frac{b}{c}*(n+1)+f(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
nm-f(c,c-b-1,a,m-1)\qquad else\\
\end{cases}\right.
$$

```c++
ll SUM(ll k)
{
    return k * (k + 1) / 2;
}

ll f(ll a,ll b,ll c,ll n)
{
    if(!a)
        return 0;
    if (a >= c || b >= c)
        return ((a / c) * SUM(n) + (n + 1) * (b / c) + f(a % c, b % c, c, n));
    ll m = (a * n + b) / c;
    return m * n - f(c, c - b - 1, a, m - 1);
}
```


$$
g(a,b,c,n)=\sum_{i=0}^{i\le n}i\left\lfloor\frac{ai+b}{c}\right\rfloor\\
$$

$$
g(a,b,c,n)=\left.\begin{cases}
\frac{a}{c}*\frac{n(n+1)(2n+1)}{6}+\frac{b}{c}*\frac{n(n+1)}{2}+g(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
\frac{nm(n+1)-f(c,c-b-1,a,m-1)-h(c,c-b-1,a,m-1)}{2}\qquad else\\
\end{cases}\right.
$$


$$
h(a,b,c,n)=\sum_{i=0}^{i\le n}\left\lfloor\frac{ai+b}{c}\right\rfloor^2\\
$$

$$
h(a,b,c,n)=\left.\begin{cases}
(\frac{a}{c})^2*\frac{n(n+1)(2n+1)}{6}+(\frac{b}{c})^2*(n+1)+(\frac{a}{c})(\frac{b}{c})*n(n+1)+h(a\%c,b\%c,c,n)+\\2*(\frac{a}{c})^2*g(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
nm(m+1)-2g(c,c-b-1,a,m-1)-2f(c,c-b-1,a,m-1)-f(a,b,c,n)\qquad else\\
\end{cases}\right.
$$

##   序列中互质数个数

$$
Comprime(x)=\sum_{d|x}{\mu(d)*cnt_d}
$$

```c++
const int maxn = 1e5 + 10;
int mu[maxn];
int cnt[maxn];
void init()
{
    for (int i = 1; i < maxn; ++i)
    {
        for (int j = i; j < maxn; j += i)
            d[j].pb(i);
        if (i == 1)
            mu[i] = 1;
        else if (i % (1ll * d[i][1] * d[i][1]) == 0)
            mu[i] = 0;
        else
            mu[i] = -mu[i / d[i][1]];
    }
}

void update(int x, int v)
{
    for(int k:d[x])
        cnt[k] += v;
}
 
int cal(int x)
{
    int res = 0;
    for(int y:d[x])
        res += mu[y] * cnt[y];
    return res;
}
```

##  积性函数

$$
\epsilon(x)=\sum_{d|x}\mu(d)\\
\epsilon(x)=\left.\begin{cases}
1\qquad x=1\\
0\qquad else
\end{cases}\right.\\
ID(x)=\sum_{d|x}\phi(d)=x\\
\phi(x)=\sum_{d|x}ID(d)\mu(\frac{x}{d})=\sum_{d|x}d\mu(\frac{x}{d})\\
$$

###  欧拉筛积性函数

$$
t[i]表示i的最小质因子出现的次数，因为欧拉筛始终用最小质因子来筛\\当i\%prime[j]=0时prime[j]自然为i的最小质因子，也是i*prime[j]的最小质因子\\
当i\%prime[j]\ne 0时prime[j]是i*prime[j]的最小质因子
$$



```c++
const int maxn = 1e5 + 10;
int prime[maxn],tot;
bool notprime[maxn];
int mu[maxn];
int t[maxn];
ll d[maxn];
void init()
{
    mu[1] = 1;
    d[1] = 1;
    for (ll i = 2; i < maxn; ++i)
    {
        if(!notprime[i])
        {
            prime[tot++] = i;
            mu[i] = -1;
            d[i] = 2;
            t[i] = 1;
        }
        for (int j = 0; j < tot && i * prime[j] < maxn; ++j)
        {
            notprime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
            {
                mu[i * prime[j]] = 0;
                d[i * prime[j]] = d[i] / (t[i] + 1) * (t[i] + 2);
                t[i * prime[j]] = t[i] + 1;
                break;
            }
            d[i * prime[j]] = d[i] * 2;
            mu[i * prime[j]] = -mu[i];
            t[i * prime[j]] = 1;
        }
    }
    for (int i = 1; i < maxn; ++i)
        mu[i] += mu[i - 1], d[i] += d[i - 1];
}
```



##  杜教筛

$$
g(x)为积性函数\\
f(x)=\sum_{i=1}^{x}g(i)\\
S(n)=\sum_{i=1}^{n}\sum_{d|i}g(d)=\sum_{d=1}^{n}\sum_{k=1}^{\left\lfloor\frac{n}{d}\right\rfloor}g(k)=\sum_{d=1}^nf(\left\lfloor\frac{n}{d}\right\rfloor)\\
f(x)=S(x)-\sum_{d=2}^nf(\left\lfloor\frac{n}{d}\right\rfloor)
$$



## min_25

$$
g(n,x)=\begin{cases}
g(n,x-1)&p_x^2\gt n\\
g(n,x-1)-f(p_x)\times\left[g(\left\lfloor\frac{n}{p_x}\right\rfloor,x-1)-\sum_{i\lt x}f(p_i)\right]&p_x^2\le n\\
\end{cases}
$$

$S(n,x)=\sum{j=1}^n[min_j\gt p_x]f(j)$
$$
S(n,x)=g(n,\infty)-\sum_{i\le x}f(i)+\sum_{k\gt x}\sum_{p_k^e\le n}f(p_k^e)\times\left[S(\left\lfloor\frac{n}{p_k^e}\right\rfloor,k)+[e\gt 1]\right]
$$

```c++
ll n, sqr;
ll inv2 = 500000004, inv3 = 333333336;
const int maxn = 1e6 + 10;
ll Prime[maxn], tot;
bool notPrime[maxn];
int ref1[maxn], ref2[maxn];//因为n/pi只有根号个取值，映射一下
ll g1[maxn], g2[maxn];//一次项函数值和二次项函数值
ll sp1[maxn], sp2[maxn];//一次函数质数项前缀和，二次函数质数项前缀和
ll w[maxn];

ll qmod(ll a,ll n)
{
	ll ans = 1;
	while(n)
	{
		if(n&1)
			ans = ans * a % mod;
		a = a * a % mod;
		n >>= 1;
	}
	return ans;
}

void init(int lim)
{
    for (ll i = 2; i <= lim;++i)
    {
        if(!notPrime[i]){
            Prime[++tot] = i;
            sp1[tot] = (i + sp1[tot - 1]) % mod;
            sp2[tot] = (i * i % mod + sp2[tot - 1]) % mod;
        }
        for (int j = 1; j <= tot && i * Prime[j] <= lim; ++j)
        {
            notPrime[i * Prime[j]] = 1;
			if (i % Prime[j] == 0)
				break;
        }
    }
}
int m;

inline int ID(ll x)
{
    if (x <= sqr)
        return ref1[x];
    return ref2[n / x];
}

void initG()
{
    m = 0;
    for (ll l = 1, r; l <= n; l = r + 1)//处理函数g(n,0),g1[i]=g1(n/i,0)=g2(w[i],0),g2[i]=g2(n/i,0)=g2(w[i],0)
    {
        r = n / (n / l);
        w[++m] = n / l;
        if (w[m] <= sqr)
            ref1[w[m]] = m;
        else
            ref2[n / w[m]] = m;
		ll ret = w[m] % mod;
		g1[m] = ((1ll + ret) * ret / 2 % mod + mod - 1) % mod;
		g2[m] = (ret * (ret + 1ll) / 2 % mod * (2 * ret + 1) % mod * inv3 % mod + mod - 1) % mod;
	}
    for (int j = 1; j <= tot; ++j)
    {
        for (int i = 1; i <= m; ++i)//处理g(n,j)
        {
            if (Prime[j] * Prime[j] > w[i])
                break;
			int k = ID(w[i] / Prime[j]);
			g1[i] = g1[i] - Prime[j] * ((g1[k] - sp1[j - 1] + mod) % mod) % mod;
			g2[i] = g2[i] - Prime[j] * Prime[j] % mod * ((g2[k] - sp2[j - 1] + mod) % mod) % mod;
			g1[i] %= mod;
			g2[i] %= mod;
			if (g1[i] < 0)
                g1[i] += mod;
            if (g2[i] < 0)
                g2[i] += mod;
        }
    }
}

ll S(ll x, ll y)
{
    if (Prime[y] >= x)
        return 0;
    ll k = ID(x);
    ll ans = ((g2[k] - g1[k] + mod) % mod - (sp2[y] - sp1[y] + mod) % mod + mod) % mod;
    for (int i = y + 1; i <= tot && Prime[i] * Prime[i] <= x; ++i)
    {
        ll pe = Prime[i];
        for (ll e = 1; pe <= x; ++e, pe *= Prime[i])
        {
            ll xx = pe % mod;
			ans = (ans + xx * (xx - 1) % mod * (S(x / pe, i) + (e != 1)) % mod) % mod;
		}
    }
    return ans % mod;
}
```



## factor类

```c++
namespace Factor {
	const int N=1010000;
	ll C,fac[10010],n,mut,a[1001000];
	int T,cnt,i,l,prime[N],p[N],psize,_cnt;
	ll _e[100],_pr[100];
	vector<ll> d;
	inline ll mul(ll a,ll b,ll p) {
		if (p<=1000000000) return a*b%p;
		else if (p<=1000000000000ll) return (((a*(b>>20)%p)<<20)+(a*(b&((1<<20)-1))))%p;
		else {
			ll d=(ll)floor(a*(long double)b/p+0.5);
			ll ret=(a*b-d*p)%p;
			if (ret<0) ret+=p;
			return ret;
		}
	}
	void prime_table(){
		int i,j,tot,t1;
		for (i=1;i<=psize;i++) p[i]=i;
		for (i=2,tot=0;i<=psize;i++){
			if (p[i]==i) prime[++tot]=i;
			for (j=1;j<=tot && (t1=prime[j]*i)<=psize;j++){
				p[t1]=prime[j];
				if (i%prime[j]==0) break;
			}
		}
	}
	void init(int ps) {
		psize=ps;
		prime_table();
	}
	ll powl(ll a,ll n,ll p) {
		ll ans=1;
		for (;n;n>>=1) {
			if (n&1) ans=mul(ans,a,p);
			a=mul(a,a,p);
		}
		return ans;
	}
	bool witness(ll a,ll n) {
		int t=0;
		ll u=n-1;
		for (;~u&1;u>>=1) t++;
		ll x=powl(a,u,n),_x=0;
		for (;t;t--) {
			_x=mul(x,x,n);
			if (_x==1 && x!=1 && x!=n-1) return 1;
			x=_x;
		}
		return _x!=1;
	}
	bool miller(ll n) {
		if (n<2) return 0;
		if (n<=psize) return p[n]==n;
		if (~n&1) return 0;
		for (int j=0;j<=7;j++) if (witness(rand()%(n-1)+1,n)) return 0;
		return 1;
	}
	ll gcd(ll a,ll b) {
		ll ret=1;
		while (a!=0) {
			if ((~a&1) && (~b&1)) ret<<=1,a>>=1,b>>=1;
			else if (~a&1) a>>=1; else if (~b&1) b>>=1;
			else {
				if (a<b) swap(a,b);
				a-=b;
			}
		}
		return ret*b;
	}
	ll rho(ll n) {
		for (;;) {
			ll X=rand()%n,Y,Z,T=1,*lY=a,*lX=lY;
			int tmp=20;
			C=rand()%10+3;
			X=mul(X,X,n)+C;*(lY++)=X;lX++;
			Y=mul(X,X,n)+C;*(lY++)=Y;
			for(;X!=Y;) {
				ll t=X-Y+n;
				Z=mul(T,t,n);
				if(Z==0) return gcd(T,n);
				tmp--;
				if (tmp==0) {
					tmp=20;
					Z=gcd(Z,n);
					if (Z!=1 && Z!=n) return Z;
				}
				T=Z;
				Y=*(lY++)=mul(Y,Y,n)+C;
				Y=*(lY++)=mul(Y,Y,n)+C;
				X=*(lX++);
			}
		}
	}
	void _factor(ll n) {
		for (int i=0;i<cnt;i++) {
			if (n%fac[i]==0) n/=fac[i],fac[cnt++]=fac[i];}
		if (n<=psize) {
			for (;n!=1;n/=p[n]) fac[cnt++]=p[n];
			return;
		}
		if (miller(n)) fac[cnt++]=n;
		else {
			ll x=rho(n);
			_factor(x);_factor(n/x);
		}
	}
	void dfs(ll x,int dep) {
		if (dep==_cnt) d.pb(x);
		else {
			dfs(x,dep+1);
			for (int i=1;i<=_e[dep];i++) dfs(x*=_pr[dep],dep+1);
		}
	}
	void norm() {
		sort(fac,fac+cnt);
		_cnt=0;
		rep(i,0,cnt) if (i==0||fac[i]!=fac[i-1]) _pr[_cnt]=fac[i],_e[_cnt++]=1;
			else _e[_cnt-1]++;
	}
	vector<ll> getd() {
		d.clear();
		dfs(1,0);
		return d;
	}
	vector<ll> factor(ll n) {
		cnt=0;
		_factor(n);
		norm();
		return getd();
	}
	vector<PLL> factorG(ll n) {
		cnt=0;
		_factor(n);
		norm();
		vector<PLL> d;
		rep(i,0,_cnt) d.pb(mp(_pr[i],_e[i]));
		return d;
	}
	bool is_primitive(ll a,ll p) {
		assert(miller(p));
		vector<PLL> D=factorG(p-1);
		rep(i,0,SZ(D)) if (powl(a,(p-1)/D[i].fi,p)==1) return 0;
		return 1;
	}
}
```

## 快速乘

```c++
inline ll mul(ll a,ll b,ll p)
{
    if (p<=1000000000)
        return a * b % p;
    else if (p<=1000000000000ll)
        return (((a * (b >> 20) % p) << 20) + (a * (b & ((1 << 20) - 1)))) % p;
    else {
        ll d=(ll)floor(a*(long double)b/p+0.5);
        ll ret=(a*b-d*p)%p;
        if (ret<0) ret+=p;
        return ret;
    }
}
```



## Miller Rabin

```c++
inline ll qmod(ll a, ll n, ll p)
{
    ll ans = 1;
    while(n)
    {
        if(n&1)
            ans = mul(ans, a, p);
        a = mul(a, a, p);
        n >>= 1;
    }
    return ans;
}

bool millerRabin(ll n)
{
    if (n < 3)
        return n == 2;
    ll a = n - 1, b = 0;
    for (; ~a & 1; a >>= 1, ++b);
    for (int i = 1, j; i <= 8; ++i)
    {
        ll x = 2 + rand() % (n - 2), v = qmod(x, a, n);
        if (v == 1 || v == n - 1)
            continue;
        for (j = 0; j < b; ++j)
        {
            v = mul(v, v, n);
            if(v==n-1)break;
        }
        if (j >= b)
            return 0;
    }
    return 1;
}
```

## Pollard rho

```c++
inline ll Pollard_rho(ll x)
{
    ll s = 0, t = 0, c = 1 + 1ll * rand() * rand() * rand() * rand() % (x - 1);
    int stp = 0, goal = 1;
    ll val = 1;
    for (goal = 1;; goal <<= 1, s = t, val = 1)
    {
        for (stp = 1; stp <= goal; ++stp)
        {
            t = (mul(t, t, x) + c) % x;
            val = mul(val, abs(t - s), x);
            if ((stp % 127) == 0)
            {
                ll d = gcd(val, x);
                if(d>1)
                    return d;
            }
        }
        ll d = gcd(val, x);
        if(d>1)
            return d;
    }
}
map<ll,ll> fac;//分解质因数
void factor(ll x)
{
    if(x<2)return;
    if(millerRabin(x))
    {
        fac[x]++;
        return;
    }
    ll p=x;
    while(p>=x)
        p=Pollard_rho(p);
    slove(p),slove(x/p);
}
```

## lucas

```c++
ll lucas(ll n, ll m, ll p)
{
    if (m == 0)
        return 1;
    return C(n % p, m % p, p) * lucas(n / p, m / p, p) % p;
}
```

## 扩展lucas

```c++
inline ll F(ll n,ll P,ll PK){
    if (n==0) return 1;
    ll rou=1;//循环节
    ll rem=1;//余项 
    for (ll i=1;i<=PK;i++){
        if (i%P) rou=rou*i%PK;
    }
    rou=qmod(rou,n/PK,PK);
    for (ll i=PK*(n/PK);i<=n;i++){
        if (i%P) rem=rem*(i%PK)%PK;
    }
    return F(n/P,P,PK)*rou%PK*rem%PK;
}
inline ll G(ll n,ll P){
    ll ans=0;
    while(n>=P)
    {
        ans+=n/P;
        n/=P;
    }
    return ans;
}
inline ll C_PK(ll n,ll m,ll P,ll PK){
    ll fz=F(n,P,PK),fm1=inv(F(m,P,PK),PK),fm2=inv(F(n-m,P,PK),PK);
    ll mi=qmod(P,G(n,P)-G(m,P)-G(n-m,P),PK);
    return fz*fm1%PK*fm2%PK*mi%PK;
}
ll A[1001],B[1001];
//x=B(mod A)
inline ll exLucas(ll n,ll m,ll P){
    ll ljc=P,tot=0;
    for (ll tmp=2;tmp*tmp<=P;tmp++){
        if (!(ljc%tmp)){
            ll PK=1;
            while (!(ljc%tmp)){
                PK*=tmp;ljc/=tmp;
            }
            A[++tot]=PK;B[tot]=C_PK(n,m,tmp,PK);
        }
    }
    if (ljc!=1){
        A[++tot]=ljc;B[tot]=C_PK(n,m,ljc,ljc);
    }
    ll ans=0;
    for (ll i=1;i<=tot;i++){
        ll M=P/A[i],T=inv(M,A[i]);
        ans=(ans+B[i]*M%P*T%P)%P;
    }
    return ans;
}
```



## BSGS

有$a^x\equiv b(\mod p)$,其中$\gcd(a,p)=1$,求$x$

令$x=A\lceil\sqrt{p}\rceil-B$,其中$0\le A,B\le \lceil p \rceil$,则有$a^{A\lceil\sqrt{p}\rceil-B}\equiv B(\mod p)$,即有$a^{A\lceil\sqrt{p}\rceil}\equiv Ba^{B}(\mod p)$

预处理出所有$a^{A\lceil p\rceil} \mod{p}$,枚举$B$

最后$x$的取值为$A\lceil\sqrt{p}\rceil-B$

#  多项式

## 高斯消元

```c++
const double eps = 1e-8;
const int maxn = 110;
double a[maxn][maxn];
bool gass(int n)
{
    for (int i = 0; i < n; ++i)
    {
        int pov = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(a[j][i]) > fabs(a[pov][i]))
                pov = j;
        for (int j = 0; j <= n; ++j)
            swap(a[i][j], a[pov][j]);
        if (fabs(a[i][i]) <= eps)
            return 0;
        for (int j = 0; j < n; ++j)
        {
            if (j != i)
            {
                double ret = a[j][i] / a[i][i];
                for (int k = i + 1; k <= n; ++k)
                    a[j][k] -= a[i][k] * ret;
            }
        }
    }
    for (int i = 0; i < n; ++i)
        a[i][n] /= a[i][i];
    return 1;
}
```



##  拉格朗日插值

$$
f(k)=\sum_{i=1}^{n}y_i\prod_{j\ne i}\frac{k-x[j]}{x[i]-x[j]}
$$

**n次多项式前缀和是n+1次多项式**

###  模板

$$
已知[f(0),f(n)]求f(k)
$$



```c++
const int maxn = 1e3 + 10;
ll f[maxn];
ll pre[maxn];
ll suf[maxn];
ll fac[maxn], infac[maxn];
void init()
{
    fac[0] = 1;
    for (int i = 1; i < maxn; ++i)
    {
        fac[i] = fac[i - 1] * i % mod;
    }
    infac[maxn - 1] = qmod(fac[maxn - 1], mod - 2);
    for (int i = maxn - 2; i >= 0; --i)
    {
        infac[i] = infac[i + 1] * (i + 1) % mod;
    }
}
ll cal(int n, int k)
{
    pre[0] = 1;
    suf[n] = 1;
    for (int i = 1; i <= n; ++i)
        pre[i] = pre[i - 1] * (k - i + 1) % mod;
    for (int i = n; i >= 1; --i)
        suf[i - 1] = suf[i] * (k - i) % mod;
    ll res = 0;
    for (int i = 0; i <= n; ++i)
    {
        ll ret = f[i] * pre[i] % mod * suf[i] % mod * infac[i] % mod * infac[n - i] % mod;
        if((n-i)&1)
            res = (res - ret + mod) % mod;
        else
            res = (res + ret) % mod;
    }
    return res;
}
```



##  虚数类

```c++
const double PI = acos(-1.0);
struct comp
{
    double r,i;
    comp(double _r = 0.0,double _i = 0.0)
    {
        r = _r; i = _i;
    }
    comp operator +(const comp &b)
    {
        return comp(r+b.r,i+b.i);
    }
    comp operator -(const comp &b)
    {
        return comp(r-b.r,i-b.i);
    }
    comp operator *(const comp &b)
    {
        return comp(r*b.r-i*b.i,r*b.i+i*b.r);
    }
};
```



##  快速傅里叶变换(FFT)

```c++
const int maxn = 4e5 + 10;
comp a[maxn], b[maxn];
char s1[maxn], s2[maxn];
int ans[maxn];
int r[maxn];
void init(int len)
{
    int l = 0;
    while ((1 << l) < len)
        ++l;
    for (int i = 0; i < len; ++i)
    {
        r[i] = (r[i >> 1] >> 1) | ((i & 1) << (l - 1));
    }
}

void DFT(comp y[],int len,int rev)//rev=1为DFT；-1为IDFT，逆变换；
{
    for (int i = 0; i < len; ++i)
    {
        if(i<r[i])
            swap(y[i], y[r[i]]);
    }
    for(int h = 1; h < len; h <<= 1)
    {
        comp wn(cos(PI / h), rev * sin(PI / h));
        for (int d = (h << 1), j = 0; j < len; j += d)
        {
            comp w(1,0);
            for (int k = j; k < j + h; k++)
            {
                comp u = y[k];
                comp t = w * y[k + h];
                y[k] = u + t;
                y[k + h] = u - t;
                w = w * wn;
            }
        }
    }
    if (rev == -1)
        for(int i = 0;i < len;i++)
            y[i].r /= len;
}
```

##  快速数论变换(NTT)

```c++
void NTT(ll y[],int len,int rev)
{
    for (int i = 0; i < len; ++i)
    {
        if(i<r[i])
            swap(y[i], y[r[i]]);
    }
    for(int h = 2; h <= len; h <<= 1)
    {
        ll wn = qmod(3, (mod - 1) / h);
        for (int j = 0; j < len; j += h)
        {
            ll w = 1;
            for (int k = j; k < j + h / 2; k++)
            {
                ll u = y[k];
                ll t = (w * y[k + h / 2]) % mod;
                y[k] = (u + t) % mod;
                y[k + h / 2] = (u - t + mod) % mod;
                w = (w * wn) % mod;
            }
        }
    }
    if (rev == -1)
    {
        reverse(y + 1, y + len);
        ll inv = qmod(len, mod - 2);
        for(int i = 0;i < len;i++)
            (y[i] *= inv) %= mod;
    }
}
```

## 快速沃尔什变换(FWT)

**按位与**
$$
FWT[A]=MERGE(FWT[A_0],FWT[A_1]+FWT[A_0])\\
UFWT[A]=MERGE(FWT[A_0],FWT[A_1]-FWT[A_0])
$$

```c++
inline void fwtor(ll a[], int len, int inv)
{
	for (int l = 1; (l<<1) <= len; l<<=1)
	{
		for (int j = l << 1, i = 0; i < len; i += j)
		{
			for (int p = 0; p < l; ++p)
			{
				a[i + l + p] += a[i + p] * inv + mod;
				a[i + l + p] %= mod;
			}
		}
	}
}
```



**按位或**
$$
FWT[A]=MERGE(FWT[A_0]+FWT[A_1],FWT[A_1])\\
UFWT[A]=MERGE(FWT[A_0]-FWT[A_1],FWT[A_1])
$$

```c++
inline void fwtand(ll a[], int len, int inv)
{
	for (int l = 1; (l<<1) <= len; l<<=1)
	{
		for (int j = l << 1, i = 0; i < len; i += j)
		{
			for (int p = 0; p < l; ++p)
			{
				a[i + p] += a[i + l + p] * inv + mod;
				a[i + p] %= mod;
			}
		}
	}
}
```



**按位异或**
$$
FWT[A]=MERGE(FWT[A_0]+FWT[A_1],FWT[A_0]-FWT[A_1])\\
UFWT[A]=MERGE(\frac{FWT[A_0]+FWT[A_1]}2,\frac{FWT[A_0]-FWT[A_1]}2)
$$

```c++
inline void fwtxor(ll a[], int len, int inv)
{
	if (inv == -1)
		inv = inv2;
	for (int l = 1; (l<<1) <= len; l<<=1)
	{
		for (int j = l << 1, i = 0; i < len; i += j)
		{
			for (int p = 0; p < l; ++p)
			{
				a[i + p] += a[i + l + p];
				a[i + l + p] = (a[i + p] - (a[i + l + p] << 1) % mod) % mod;

				a[i + p] *= inv;
				a[i + p] %= mod;
				if (a[i + p] < 0)
					a[i + p] += mod;

				a[i + l + p] *= inv;
				a[i + l + p] %= mod;
				if (a[i + l + p] < 0)
					a[i + l + p] += mod;
			}
		}
	}
}
```



**按位同或**
$$
FWT[A]=MERGE(FWT[A_1]-FWT[A_0],FWT[A_1]+FWT[A_0])\\
UFWT[A]=MERGE(\frac{FWT[A_1]-FWT[A_0]}2,\frac{FWT[A_1]+FWT[A_0]}2)
$$

```c++
inline void fwtxor(ll a[], int len, int inv)
{
	if (inv == -1)
		inv = inv2;
	for (int l = 1; (l<<1) <= len; l<<=1)
	{
		for (int j = l << 1, i = 0; i < len; i += j)
		{
			for (int p = 0; p < l; ++p)
			{
				a[i + l + p] += a[i + p];
				a[i + p] = (a[i + l + p] - (a[i + p] << 1) % mod ) %mod;

				a[i + p] *= inv;
				a[i + p] %= mod;
				if (a[i + p] < 0)
					a[i + p] += mod;

				a[i + l + p] *= inv;
				a[i + l + p] %= mod;
				if (a[i + l + p] < 0)
					a[i + l + p] += mod;
			}
		}
	}
}
```



## 多项式求逆

```c++
ll temp[maxn];
void inv(ll a[], ll inva[], ll n)
{
    ll lim = 1;
    while (lim < n)
        lim <<= 1;
    inva[0] = qmod(a[0], mod - 2);
    for (ll len = 1; len <= lim; len <<= 1)
    {
        init(len << 1);
        for (int i = 0; i < len; ++i)
        {
            temp[i] = a[i];
            temp[i + len] = 0;
        }
        NTT(temp, len << 1, 1);
        NTT(inva, len << 1, 1);
        for (int i = 0; i < (len << 1); ++i)
            inva[i] = (2ll - temp[i] * inva[i] % mod + mod) % mod * inva[i] % mod;
        NTT(inva, len << 1, -1);
        for (int i = 0; i < len; ++i)
            inva[i + len] = 0;
    }
}

```

## 多项式除法|取模

```c++
void mul(ll x[], int n, ll y[], int m, ll res[])
{
    int len = 1;
    while (len < n + m - 1)
        len <<= 1;
    for (int i = n; i < len; ++i)
        x[i] = 0;
    for (int i = m; i < len; ++i)
        y[i] = 0;
    init(len);
    NTT(x, len, 1);
    NTT(y, len, 1);
    for (int i = 0; i < len; ++i)
        res[i] = x[i] * y[i] % mod;
    NTT(res, len, -1);
}
ll temp1[maxn << 2], temp2[maxn << 2];
//Q*G+R=F
//Q: n-m+1,R: m-1
void div(ll F[], int n, ll G[], int m, ll Q[], ll R[])
{
    for (int i = 0; i < n; ++i)
        temp1[i] = F[i];
    reverse(temp1, temp1 + n);
    reverse(G, G + m);
    inv(G, temp2, n - m + 1);
    reverse(G, G + m);
    mul(temp1, n, temp2, n - m + 1, Q);
    reverse(Q, Q + n - m + 1);
    for (int i = 0; i < n - m + 1; ++i)
        temp1[i] = Q[i];
    for (int i = 0; i < m; ++i)
        temp2[i] = G[i];
    mul(temp1, n - m + 1, temp2, m, R);
    for (int i = 0; i < m - 1; ++i)
        R[i] = (F[i] - R[i] + mod) % mod;
}
```

## 线性递推

线性递推可用

k项线性递推求第 n项,复杂度
$$
O(k^2\log(n))
$$

```c++

namespace linear_seq {
    const int N=10010;
    ll res[N],base[N],_c[N],_md[N];
 
    vector<int> Md;
    void mul(ll *a,ll *b,int k) {
        rep(i,0,k+k) _c[i]=0;
        rep(i,0,k) if (a[i]) rep(j,0,k) _c[i+j]=(_c[i+j]+a[i]*b[j])%mod;
        for (int i=k+k-1;i>=k;i--) if (_c[i])
            rep(j,0,SZ(Md)) _c[i-k+Md[j]]=(_c[i-k+Md[j]]-_c[i]*_md[Md[j]])%mod;
        rep(i,0,k) a[i]=_c[i];
    }
    int solve(ll n,VI a,VI b) {  /// a 系数 b 初值 b[n+1]=a[0]*b[n]+...
        ll ans=0,pnt=0;
        int k=SZ(a);
        assert(SZ(a)==SZ(b));
        rep(i,0,k) _md[k-1-i]=-a[i];_md[k]=1;
        Md.clear();
        rep(i,0,k) if (_md[i]!=0) Md.push_back(i);
        rep(i,0,k) res[i]=base[i]=0;
        res[0]=1;
        while ((1ll<<pnt)<=n) pnt++;
        for (int p=pnt;p>=0;p--) {
            mul(res,res,k);
            if ((n>>p)&1) {
                for (int i=k-1;i>=0;i--) res[i+1]=res[i];res[0]=0;
                rep(j,0,SZ(Md)) res[Md[j]]=(res[Md[j]]-res[k]*_md[Md[j]])%mod;
            }
        }
        rep(i,0,k) ans=(ans+res[i]*b[i])%mod;
        if (ans<0) ans+=mod;
        return ans;
    }
    VI BM(VI s) {
        VI C(1,1),B(1,1);
        int L=0,m=1,b=1;
        rep(n,0,SZ(s)) {
            ll d=0;
            rep(i,0,L+1) d=(d+(ll)C[i]*s[n-i])%mod;
            if (d==0) ++m;
            else if (2*L<=n) {
                VI T=C;
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                L=n+1-L; B=T; b=d; m=1;
            } else {
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                ++m;
            }
        }
        return C;
    }
    int gao(VI a,ll n) {//下标从0开始
        VI c=BM(a);
        c.erase(c.begin());
        rep(i,0,SZ(c)) c[i]=(mod-c[i])%mod;
        return solve(n,c,VI(a.begin(),a.begin()+SZ(c)));
    }
};
```



# 组合数学

## 卡特兰数

+   给出2n个元素，其中n个A元素，n个B元素，要求对于由两种元素按顺序排列组成的序列中，任意一个前缀B的数量不少于A,有多少种组合方式
+   n个节点的二叉树共有多少个
    +   可以将一棵二叉树分为根，左子树，右子树三部分，枚举左子树节点个数

$$
C(n)=\frac{C_{2n}^{n}}{n+1}\\
C(n)=\sum_{i=0}^{n-1}C(i)\times C(n-i-1)\\
C(n+1)=\frac{4n+2}{n+2}C(n)\\
C(n)=C_{2n}^{n}-C_{2n}^{n-1}
$$

## 斯特林数

### 第一类斯特林数

**把 n个不同的球排成 r个非空循环排列的方法数**
$$
S(n,r)=(n-1)S(n-1,r)+s(n-1,r-1)
$$
考虑最后一个球，它可以单独构成一个非空循环排列，也可以插入到前面的某一个球的一侧。

若单独放，则有**S(n-1,r-1)** 种放法；若放在某个球的一侧，则有 **(n-1)S(n-1,r)**种放法。

**S(n,r)**为多项式
$$
f_n(x)=\prod_{i=0}^{n-1}(x+i)
$$
的k次项系数，分治法+FFT/NTT可求

### 第二类斯特林数

**把n个不同的球放到r个相同的盒子里，方案数**
$$
S(n,r)=rS(n-1,r)+S(n-1,r-1)\\
S(n,r)=\frac{1}{r!}\sum_{k=0}^{r}(-1)^kC_{r}^{k}(r-k)^n\\
n^r=\sum_{i=0}^{r}S(r,i)\times i!\times C_{n}^{i}\\
$$
卷积形式
$$
\begin{align}S(n,m)&=\frac{1}{m!}\sum_{k=0}^{m}(-1)^k\frac{m!}{k!(m-k)!}(m-k)^n\\
&=\sum_{k=0}^{m}\frac{(-1)^k}{k!}\frac{(m-k)^n}{(m-k)!}\\
\end{align}
$$
令
$$
f(x)=\frac{(-1)^x}{x!}\\
g_n(x)=\frac{(m-k)^n}{(m-k)!}
$$
则
$$
S(n,m)=\sum_{k=0}^kf(k)\times g_n(m-k)\\
$$
卷积形式，NTT求解

# 字符串

##  后缀数组

```c++
const int maxn = 1e5 + 10;
char s[maxn];
int x[maxn], y[maxn], sa[maxn], c[maxn], height[maxn], rak[maxn];//rak[i],后缀i在排完序的数组里的排名，sa[i]排名为i的后缀在原串里的位置
int st[maxn][20];
int n, m;
inline void getsa()
{
    int *xx=x,*yy=y;
	memset(c, 0, sizeof c);
	for (int i = 1; i <= n; ++i)++c[xx[i] = s[i]];
	for (int i = 2; i <= m; ++i)c[i] += c[i - 1];
	for (int i = n; i >= 1; --i)sa[c[xx[i]]--] = i;
	for (int k = 1; k <= n; k <<= 1)
	{
		int num = 0;
		for (int i = n - k + 1; i <= n; ++i)yy[++num] = i;
		for (int i = 1; i <= n; ++i)if (sa[i] > k)yy[++num] = sa[i] - k;
		for (int i = 1; i <= m; ++i)c[i] = 0;
		for (int i = 1; i <= n; ++i)++c[xx[i]];
		for (int i = 2; i <= m; ++i)c[i] += c[i - 1];
		for (int i = n; i >= 1; --i)sa[c[xx[yy[i]]]--] = yy[i], yy[i] = 0;//sort
		swap(xx, yy);
		xx[sa[1]] = 1; num = 1;
		for (int i = 2; i <= n; ++i)
			xx[sa[i]] = (yy[sa[i]] == yy[sa[i - 1]] && yy[sa[i] + k] == yy[sa[i - 1] + k]) ? num : ++num;
		if (num == n)break;
		m = num;
	}
}

inline void getheight()
{
	int k = 0;
	for (int i = 1; i <= n; ++i)rak[sa[i]] = i;
	for (int i = 1; i <= n; ++i)
	{
		if (rak[i] == 1)continue;
		if (k)--k;
		int j = sa[rak[i] - 1];
		while (j + k <= n && i + k <= n && s[i + k] == s[j + k])++k;
		height[rak[i]] = k;
	}
}

inline void buildst(int n)
{
	for (int i = 1; i <= n; ++i)st[i][0] = height[i];
	for (int k = 1; k < 20; ++k)
	{
		for (int i = 1; i + (1 << k) - 1 <= n; ++i)
			st[i][k] = min(st[i][k - 1], st[i + (1 << k - 1)][k - 1]);
	}
}

inline int lcp(int x, int y)
{
	int l = rak[x], r = rak[y];
	if (l > r)swap(l, r);
	if (l == r)return n - sa[l] + 1;
	int t = log2(r - l);
	return min(st[l + 1][t], st[r - (1 << t) + 1][t]);
}
```

##  后缀自动机(SAM)

计算子串出现次数：对每个endpos大小进行累加

endpos大小通过后缀链接，对每个不是复制的节点进行累加

计算不同字串转移数量：对每个状态进行累加

```c++
struct NODE
{
	int ch[26];
	int len, fa;
	NODE() { memset(ch, 0, sizeof(ch)); len = 0; }
    // inline void clear()
    // {
    //     memset(ch, 0, sizeof ch);
    //     len = fa = 0;
    // }
}node[maxn << 1];
int las = 1, tot = 1;

void init()//初始化
{
    las = tot = 1;
    node[1].clear();
}

void add(int c)
{
	int p = las; int np = las = ++tot;
    //node[np].clear();//用到时再清空，只需要初始化node[1]既可以快速初始化
	node[np].len = node[p].len + 1;
	for (; p && !node[p].ch[c]; p = node[p].fa)node[p].ch[c] = np;
	if (!p)node[np].fa = 1;//以上为case 1
	else
	{
		int q = node[p].ch[c];
		if (node[q].len == node[p].len + 1)node[np].fa = q;//以上为case 2
		else
		{
			int nq = ++tot; node[nq] = node[q];
			node[nq].len = node[p].len + 1;
			node[q].fa = node[np].fa = nq;
			for (; p && node[p].ch[c] == q; p = node[p].fa)node[p].ch[c] = nq;//以上为case 3
		}
	}
}
//注释部分为广义后缀自动机
/*struct SAM
{
	int ch[maxn << 1][26];
	int fa[maxn << 1], len[maxn << 1];
	int las = 1, tot = 1;
	void init()
	{
		memset(ch[1], 0, sizeof ch[1]);
		fa[1] = len[1] = 0;
		las = tot = 1;
	}

	void clear(int p)
	{
		memset(ch[p], 0, sizeof ch[0]);
		len[p] = fa[p] = 0;
	}

	void add(int c)//,int las)
	{
        //if(ch[las][c]&&len[las]+1==len[ch[las][c]])return ch[las][c];
		int p = las; int np = las = ++tot;
		//bool flag=0;
		// clear(np);
		len[np] = len[p] + 1;
		for (; p && !ch[p][c]; p = fa[p])ch[p][c] = np;
		if (!p)fa[np] = 1;
		else
		{
			int q = ch[p][c];
			if (len[q] == len[p] + 1)fa[np] = q;
			else
			{
            	//if(len[p]+1==len[np])flag=1;
				int nq = ++tot;
				memcpy(ch[nq], ch[q], sizeof ch[nq]);
				fa[nq] = fa[q];
				len[nq] = len[p] + 1;
				fa[q] = fa[np] = nq;
				for (; p && ch[p][c] == q; p = fa[p])ch[p][c] = nq;
			}
		}
		//return flag?nq:np;
	}
	int c[maxn << 1], b[maxn << 1];
	inline void sort()
	{
		for (int i = 1; i <= tot; ++i)++c[len[i]];
		for (int i = 1; i <= tot; ++i)c[i] += c[i - 1];
		for (int i = 1; i <= tot; ++i)b[c[len[i]]--] = i;
	}
};*/
/*
	for(int t=nextint();t--;)
	{
		int las=1;
		for(int i=0;s[i];++i)
		{
			las=sam.add(s[i]-'a',las);
		}
	}
*/

char s[MAXN]; int len;
int main()
{
	scanf("%s", s); len = strlen(s);
	for (int i = 0; i < len; i++)add(s[i] - 'a');
}
```

## AC自动机

```c++
const int maxn = 2e4 + 10;
const int maxm = 1e6 + 10;
int trie[maxn << 1][26];
int fail[maxn << 1];
int cntword[maxn << 1];
int ans[maxn];
int cnt = 0;
char s[200][100], t[maxm];
void insertwords(char str[], int idx)
{
	int u = 0;
	for (int i = 0; str[i]; ++i)
	{
		int v = str[i] - 'a';
		if (!trie[u][v])
			trie[u][v] = ++cnt;
		u = trie[u][v];
	}
	cntword[u] = idx;
}

void getfail()
{
	queue<int> q;
	for (int i = 0; i < 26; ++i)if (trie[0][i])
	{
		fail[trie[0][i]] = 0;
		q.push(trie[0][i]);
	}
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		for (int i = 0; i < 26; ++i)
		{
			if (trie[u][i])
			{
				fail[trie[u][i]] = trie[fail[u]][i];
				q.push(trie[u][i]);
			}
			else
				trie[u][i] = trie[fail[u]][i];
		}
	}
}
int maxx = 0;
int query(char str[])
{
	int u = 0, res = 0;
	for (int i = 0; str[i]; ++i)
	{
		u = trie[u][str[i] - 'a'];
		for (int j = u; j; j = fail[j])
		{
			++ans[cntword[j]];
			if (cntword[j])
				maxx = max(maxx, ans[cntword[j]]);
		}
	}
	return res;
}

int main()
{
	int n;
	while (scanf("%d", &n), n)
	{
		cnt = 0;
		maxx = 0;
		memset(cntword, 0, sizeof cntword);
		memset(trie, 0, sizeof trie);
		memset(fail, 0, sizeof fail);
		memset(ans, 0, sizeof ans);
		for (int i = 1; i <= n; ++i)
		{
			scanf("%s", s[i]);
			insertwords(s[i], i);
		}
		scanf("%s", t);
		fail[0] = 0;
		getfail();
		query(t);
		printf("%d\n", maxx);
		for (int i = 1; i <= n; ++i)if (ans[i] == maxx)
			printf("%s\n", s[i]);
	}
}
```

## 最小表示法

```C++
int zxbsf(string s)
{
	int len = s.length() - 1;
	s.append(s);
	int i = 0, j = 1, k;
	while (i <= len && j <= len)
	{
		for (k = 0; k < len && s[i + k] == s[j + k]; k++);
		if (k == len)break;
		if (s[i + k] > s[j + k])
		{
			i = i + k + 1;
			if (i == j)i++;
		}
		else
		{
			j = j + k + 1;
			if (i == j)j++;
		}
	}
	return min(i, j);
}
```

#  图论

## 网络流

###  Dinic

```c++
const int maxm = 1e5 + 10;
const int maxn = 1e4 + 10;
struct EDGE
{
	int next, to, w;
}edge[maxm << 1];
int head[maxn], tot;
int cur[maxn];
int n, m;
inline void addedge(int u, int v, int w)
{
	edge[tot] = { head[u],v,w };
	head[u] = tot++;
	edge[tot] = { head[v],u,0 };
	head[v] = tot++;
}

int dep[maxn];

bool bfs(int s, int t)
{
	memset(dep, 0x3f, sizeof dep);
	queue<int> q;
	for (int i = 0; i <= n; ++i)cur[i] = head[i];
	dep[s] = 0;
	q.push(s);
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		for (int i = head[u]; ~i; i = edge[i].next)
		{
			int v = edge[i].to;
			if (dep[v] == inf && edge[i].w)
			{
				dep[v] = dep[u] + 1;
				q.push(v);
			}
		}
	}
	return dep[t] < inf;
}
int dfs(int u, int t, int limit)
{
	if (!limit || u == t)return limit;
	int flow = 0, f;
	for (int i = cur[u]; ~i; i = edge[i].next)
	{
		cur[u] = i;
		int v = edge[i].to;
		if (dep[v] == dep[u] + 1 && (f = dfs(v, t, min(limit, edge[i].w))))
		{
			flow += f;
			limit -= f;
			edge[i].w -= f;
			edge[i ^ 1].w += f;
			if (!limit)break;
		}
	}
	return flow;
}

ll Dinic(int s, int t)
{
	ll maxflow = 0;
	while (bfs(s, t))
		maxflow += dfs(s, t, inf);
	return maxflow;
}

```

##  最小费用最大流

###   spfa+EK

```c++
bool vis[maxn];
int dis[maxn], flow[maxn], pre[maxn], last[maxn];
bool spfa(int s, int t)
{
    queue<int> q;
    memset(dis, 0x3f, sizeof dis);
    memset(flow, 0x3f, sizeof flow);
    memset(vis, 0, sizeof vis);
    q.push(s);
    vis[s] = 1;
    dis[s] = 0;
    pre[t] = -1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (int i = head[u]; ~i; i = edge[i].next)
        {
            int v = edge[i].to;
            if (edge[i].w > 0 && dis[v] > dis[u] + edge[i].c)
            {
                dis[v] = dis[u] + edge[i].c;
                pre[v] = u;
                last[v] = i;
                flow[v] = min(flow[u], edge[i].w);
                if(!vis[v])
                {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return pre[t] != -1;
}
ll maxflow, maxcost;
void mcmf(int s, int t)
{
    maxflow = 0, maxcost = 0;
    while(spfa(s,t))
    {
        int u=t;
        maxflow += flow[t];
        maxcost += flow[t] * dis[t];
        while (u != s)
        {
            edge[last[u]].w -= flow[t];
            edge[last[u] ^ 1].w += flow[t];
            u = pre[u];
        }
    }
}
```

## 启发式搜索(K短路/A*)

```c++
const int maxn = 1e5 + 10;
int tot1 = 0, tot2 = 0;
int head1[1010], head2[1010];
int dis[1010];
bool vis[1010];
struct Edge
{
	int next, to, w;
	bool operator < (const Edge& other)const
	{
		return w + dis[to] > other.w + dis[other.to];
	}
} edge1[maxn], edge2[maxn];
struct cmp
{
	bool operator () (const Edge& a, const Edge& b)
	{
		return a.w > b.w;
	}
};
inline void addedge1(int u, int v, int w)
{
	edge1[tot1] = { head1[u],v,w };
	head1[u] = tot1++;
}
inline void addedge2(int u, int v, int w)
{
	edge2[tot2] = { head2[u],v,w };
	head2[u] = tot2++;
}
void dijk(int s, int t)
{
	memset(dis, 0x3f, sizeof dis);
	dis[s] = 0;
	priority_queue<Edge, vector<Edge>, cmp > q;
	for (int i = head2[s]; ~i; i = edge2[i].next)
	{
		int v = edge2[i].to;
		dis[v] = min(dis[v], edge2[i].w);
		q.push(edge2[i]);
	}
	Edge tmp;
	while (!q.empty())
	{
		tmp = q.top();
		q.pop();
		int u = tmp.to;
		vis[u] = true;
		for (int i = head2[u]; ~i; i = edge2[i].next)
		{
			int v = edge2[i].to, w = edge2[i].w;
			if (vis[v])continue;
			if (dis[v] > dis[u] + w)
			{
				dis[v] = dis[u] + w;
				q.push(Edge{ 0,v,dis[u] + w });
			}
		}
	}
}

int Astar(int s, int t, int k)
{
	int cnt = 0;
	priority_queue<Edge> q;
	for (int i = head1[s]; ~i; i = edge1[i].next)
	{
		q.push(edge1[i]);
	}
	Edge tmp;
	while (!q.empty())
	{
		tmp = q.top();
		q.pop();
		if (tmp.to == t)
		{
			++cnt;
			if (cnt == k)
				return tmp.w;
		}
		int u = tmp.to;
		int w = tmp.w;
		for (int i = head1[u]; ~i; i = edge1[i].next)
		{
			q.push(Edge{ 0,edge1[i].to,w + edge1[i].w });
		}
	}
	return -1;
}

int main()
{
	int n, m;
	int u, v, w;
	int s, t, k;
	memset(head1, -1, sizeof head1);
	memset(head2, -1, sizeof head2);
	read(n), read(m);
	for (int i = 0; i < m; ++i)
	{
		read(u), read(v), read(w);
		addedge1(u, v, w);
		addedge2(v, u, w);
	}
	read(s), read(t), read(k);
	dijk(t, s);
	printf("%d\n", Astar(s, t, k));
}

```

## 曼哈顿最小生成树

```c++
struct POINT
{
    int x, y, id;
} point[maxn], temp[maxn];
struct BIT
{
    int _N;
    pii bit[maxn];
    void init(int n)
    {
        _N = n;
        for (int i = 1; i <= n; ++i)
        {
            bit[i] = pii(inf, -1);
        }
    }

    void update(int k,pii v)
    {
        for (; k > 0; k -= lowbit(k))
        {
            bit[k] = min(bit[k], v);
        }
    }

    pii query (int k)
    {
        pii res(inf, -1);
        for (; k <= _N; k += lowbit(k))
        {
            res = min(res, bit[k]);
        }
        return res;
    }
} bit;

int b[maxn];
void manhattan_mst(POINT p[], int n)
{
    for (int cas = 0; cas < 4; ++cas)
    {
        if (cas == 1 || cas == 3)
            for (int i = 0; i < n; ++i)
                swap(p[i].x, p[i].y);
        else if (cas == 2)
            for (int i = 0; i < n; ++i)
                p[i].x = -p[i].x;
        for (int i = 0; i < n; ++i)
        {
            b[i] = p[i].y - p[i].x;
        }
        sort(b, b + n);
        int m = unique(b, b + n) - b;
        sort(p, p + n, [](POINT a, POINT b) {
            return a.x == b.x ? a.y > b.y : a.x > b.x;
        });
        bit.init(m);
        for (int i = 0; i < n; ++i)
        {
            int l = lower_bound(b, b + m, p[i].y - p[i].x) - b;
            pii res = bit.query(l + 1);
            if (res.second != -1)
                edge[tot++] = {p[i].id, res.second, res.first - p[i].x - p[i].y};
            bit.update(l + 1, pii(p[i].x + p[i].y, p[i].id));
        }
    }
}
```



#  计算几何

## 三点定圆

```c++
const double eps = 1e-6;
struct Point {
	double x, y;
	Point(double _x = 0, double _y = 0)
	{
		x = _x;
		y = _y;
	}
	Point operator-(const Point& p1)const
	{
		return Point(x - p1.x, y - p1.y);
	}
	Point operator+(const Point& p1)const
	{
		return Point(x + p1.x, y + p1.y);
	}
	bool friend operator<(const Point& p1, const Point& p2)
	{
		if (p1.x != p2.y)return p1.x < p2.x;
		return p1.y < p2.y;
	}
	bool friend operator==(const Point& p1, const Point& p2)
	{
		if (fabs(p1.x - p2.x) > eps)return 0;
		if (fabs(p1.y - p2.y) > eps)return 0;
		return 1;
	}
	double friend operator^(const Point& p1, const Point& p2)
	{
		return p1.x * p2.y - p2.x * p1.y;
	}
	Point friend operator*(const Point& p1, const double& k)
	{
		return Point(p1.x * k, p1.y * k);
	}
	double friend operator*(const Point& p1, const Point& p2)
	{
		return p1.x * p2.x + p1.y * p2.y;
	}
	void transXY(double B)//绕原点旋转B弧度 
	{
		double tx = x, ty = y;
		x = tx * cos(B) - ty * sin(B);
		y = tx * sin(B) + ty * cos(B);
	}
}aa[maxn];
double dis(const Point& p1, const Point& p2)
{
	return sqrt((p1 - p2) * (p1 - p2));
}
bool in_row(const Point& p1, const Point& p2, const Point& p3)//判断三点是否共线 
{
	return fabs((p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)) <= eps;
}
Point GetCenter(const Point& p1, const Point& p2, const Point& p3)//求三点圆心 （三角形外心） 
{
	Point cp;
	double a1 = p2.x - p1.x, b1 = p2.y - p1.y, c1 = (a1 * a1 + b1 * b1) / 2;
	double a2 = p3.x - p1.x, b2 = p3.y - p1.y, c2 = (a2 * a2 + b2 * b2) / 2;
	double d = a1 * b2 - a2 * b1;
	cp.x = p1.x + (c1 * b2 - c2 * b1) / d;
	cp.y = p1.y + (a1 * c2 - a2 * c1) / d;
	return cp;
}

```

## 平面最近点对

```c++
const int maxn = 1e5 + 10;
struct point
{
    double x, y;
} p[maxn];

double dis(point a,point b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
bool cmpx(point a,point b)
{
    return a.x < b.x;
}
bool cmpy(point a,point b)
{
    return a.y < b.y;
}
double cal(point a[],int len)
{
    if(len==1)
        return inf;
    if(len==2)
        return dis(a[0], a[1]);
    double p = a[len / 2 - 1].x;
    int mid = len / 2;
    double d = min(cal(a, mid), cal(a + mid, len - mid));
    int tot = 0;
    for (int i = 0; i < len; ++i)
        if (fabs(p - a[i].x) <= d)
            swap(a[tot++], a[i]);
    sort(a, a + tot, cmpy);
    for (int i = 0; i < tot; ++i)
    {
        for (int j = i + 1; j < tot; ++j)
        {
            if (a[j].y - a[i].y > d)
                break;
            d = min(d, dis(a[i], a[j]));
        }
    }
    return d;
}

int main()
{
    int n;
    read(n);
    for (int i = 0; i < n; ++i)
    {
        scanf("%lf%lf", &p[i].x, &p[i].y);
    }
    sort(p, p + n, cmpx);
    printf("%.6f\n", cal(p, n));
}
```



# 其它

## 关于括号匹配

1.  取**‘（’**为1，**‘）’**为-1，若序列每个前缀和均非负，切总和为0，则括号可成功匹配
2.  接上条，对于总和为0的匹配序列，序列中前缀和最小值的个数即为循环匹配的个数

## n个数全不在自己位置上的匹配/错排递推

$$
f(n+1)=nf(n)+nf(n-1)
$$

