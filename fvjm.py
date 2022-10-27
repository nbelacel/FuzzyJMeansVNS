import sys
import time
import numpy as np
from fvjm_init import *
import pandas as pd
import math
import copy
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fuzzy_index.cvi import *
from core.plotting import *
import random


def generate_random_color(n):
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n)]
    c_sets = {}
    for i, c in enumerate(color):
        c_sets[i + 1] = c
    return c_sets


def inite(x, n, d):
    v = np.zeros((NUM_OF_CLUSTER, d))
    for i in range(d):
        min_, max_ = x[0][i], x[0][i]
        for j in range(n):
            xji = x[j][i]
            if (xji < min_):
                min_ = xji
            elif xji > max_:
                max_ = xji
        dist_ = (max_ - min_) / (NUM_OF_CLUSTER + 1)
        for k in range(NUM_OF_CLUSTER):
            v[k][i] = min_ + dist_ * k
    return v


def Eucdist(x, y, d):
    sum_ = 0.0
    for i in range(d):
        sum_ += (x[i] - y[i]) * (x[i] - y[i])
    return math.sqrt(sum_)


def Calculate_w(n, c, x, v, m, w, wt, d):
    t = 2 / (m - 1)
    dists = np.zeros(c)
    for i in range(n):
        for j in range(c):
            wt[j][i] = 0
            w[i][j] = 0
        f = False
        for j in range(c):
            # dists[j] = pow(np.linalg.norm(x[i] - v[j]), t)
            dists[j] = pow(Eucdist(x[i], v[j], d), t)
            if dists[j] < SMALL:
                wt[j][i] = w[i][j] = 1
                f = True
                break
        if not f:
            for j in range(c):
                d1 = dists[j]
                s = 0
                for k in range(c):
                    s += d1 / dists[k]
                wt[j][i] = pow(1.0 / s, m)
                w[i][j] = 1 / s
    return w, wt


def Calculate_Newv(t, n, d, x, wt, tempv):
    # tempv = np.zeros((t,d))
    for i in range(t):
        sumd = 0
        for k in range(n):
            sumd += wt[i][k]
        for j in range(d):
            sumn = 0
            for k in range(n):
                sumn += (x[k][j] * wt[i][k])
            tempv[i][j] = sumn / sumd
    return tempv


def Unoccupied(c, n, d, x, v, ind):
    for i in range(n):
        j = 0
        for k in range(c):
            for l in range(d):
                d1 = x[i][l] - v[k][l]
                if abs(d1) > SMALL:
                    j += 1
                    break
        if j == c:
            ind[i] = True
        else:
            ind[i] = False
    return ind


def Best_deletion(c, n, d, x, v, m, sc):
    t = 1 - m
    temp = np.zeros(c)
    for l in range(c):
        dd = 0
        for i in range(n):
            d1 = Eucdist(x[i], v[l], d)
            if d1 > 0.0001:
                d1 = pow((d1 * d1), (1 / t))
            else:
                d1 = 0
            dd += pow((sc[i] - d1), t)
        temp[l] = dd
    min_ = temp[0]
    co = 0
    for i in range(c):
        if temp[i] < min_:
            min_ = temp[i]
            co = i
    return co


def Best_insertion(n, c, d, ind, z, x, v, m, sc):
    tabk = np.zeros(n)
    for ll in range(n):
        if ind[ll] == True:
            tabk[ll] = Calw(n, c, d, z, ll, x, v, m, sc)
        else:
            tabk[ll] = sys.maxsize
    min_ = tabk[0]
    f = 0
    for i in range(n):
        if tabk[i] < min_:
            min_ = tabk[i]
            f = i
    return f


def Calw(n, c, d, f, z, x, v, m, sint):
    d1, d2, d3, s = 0, 0, 0, 0
    t = 1 / (m - 1)
    for i in range(n):
        ed = Eucdist(x[i], v[f], d)
        ed = ed * ed
        d1 = pow(ed, t)
        if d1 < 0.001:
            d1 = 0
        else:
            d1 = 1 / d1
        if i == z:
            d2 = 0
        else:
            ed2 = Eucdist(x[i], x[z], d)
            ed2 = ed2 * ed2
            d2 = pow(ed2, t)
            if d2 == 0:
                d2 = sys.maxsize
            else:
                d2 = 1 / d2
            # print(d2)
        d3 = pow((sint[i] - d1 + d2), m - 1)
        if d3 < 0.001:
            d3 = 0
        else:
            d3 = 1 / d3
        s += d3
    return s


def Errorv(c, v, tempv, d):
    for i in range(c):
        # e = np.linalg.norm(tempv[i] - v[i])
        e = Eucdist(tempv[i], v[i], d)
    return e


def Newv(c, d, v, tempv):
    return tempv


def Calculate_R(n, c, x, v, m, d):
    d2 = 0
    t = m - 1
    s = Calculate_S(n, c, x, v, m, d)
    for i in range(n):
        d2 += 1 / pow(s[i], t)
    return d2


def Calculate_S(n, c, x, v, m, d):
    t = 1 / (m - 1)
    sc = np.zeros(n)
    for i in range(n):
        s = 0
        for j in range(c):
            # d1 = np.linalg.norm(x[i] - v[j])
            d1 = Eucdist(x[i], v[j], d)
            if (d1 < SMALL):
                d1 = 0
            else:
                d1 = pow((d1 * d1), t)
                d1 = 1 / d1
            s += d1
        sc[i] = s
    return sc


def fj_means(n, c, d, max_iter, E, x, v, w, wt, tempv, m, obj):
    it = 0
    objtmp = 0
    sc = np.zeros(50000)
    ind = np.zeros(n)
    tempv1 = np.zeros((c, d))
    it1 = 10000
    obj = 0
    t1 = 0
    iter_ = 0
    v, w, wt, tempv, obj, tt2 = f_cmeans(E, it, t1, n, c, x, v, w, wt, m, d, tempv, it, objtmp)
    tempv1 = copy.deepcopy(v)
    start = time.time()
    it1 = 1
    tb = 2
    while tb > 0:
        iter_ += 1
        ind = Unoccupied(c, n, d, x, tempv1, ind)
        centout = Best_deletion(c, n, d, x, tempv1, m, sc)
        centin = Best_insertion(n, c, d, ind, centout, x, tempv1, m, sc)
        for i in range(d):
            tempv1[centout][i] = x[centin][i]
        # f_cmeans(E,1,t1,n,c,x,tempv1,w,wt,m,d,tempv,it1,objtmp)
        tempv1, w, wt, tempv, objtmp, tt2 = f_cmeans(E, 1, t1, n, c, x, tempv1, w, wt, m, d, tempv, it1, objtmp)

        if objtmp < obj:
            obj = objtmp
        else:
            v = copy.deepcopy(tempv1)
            break
        sc = Calculate_S(n, c, x, tempv1, m, d)
        if iter_ > max_iter:
            break
        tb -= 1
    end_ = time.time()
    t = (end_ - start)
    it = iter_
    print("time", t)
    w, wt = Calculate_w(n, c, x, v, m, w, wt, d)
    return v, w, wt, obj


def f_cmeans(E, tt1, tt, n, c, x, v, w, wt, m, d, tempv, it, obj):
    iter_ = 0
    st = time.time()
    while True:
        iter_ += 1
        w, wt = Calculate_w(n, c, x, v, m, w, wt, d)
        tempv = Calculate_Newv(c, n, d, x, wt, tempv)
        error_ = Errorv(c, v, tempv, d)
        v = copy.deepcopy(tempv)
        if tt1 == 1:
            error_ = E
        if error_ <= E:
            break
    it = iter_
    tt = time.time() - st
    obj = Calculate_R(n, c, x, v, m, d)
    w, wt = Calculate_w(n, c, x, v, m, w, wt, d)
    return v, w, wt, tempv, obj, tt


def vns1(n, d, x, v, w, wt, tempv):
    it1, obj, t = 0, 0, 0
    c = NUM_OF_CLUSTER
    t0 = MAX_CPU_TIME
    E = TERMINATION_THRESHOLD
    m = FUZZY_FACTOR
    kmax, nbiter, tt, objtmp, tt1 = 0, 0, 0, 0, 0
    it = 1000
    sc = np.zeros(50000)
    ind = np.zeros(n)
    tempv1 = np.zeros((c, d))
    kmax = min(c, 10)
    kstep, kfirst = 1, 1
    stime = time.time()
    v, w, wt, tempv, objtmp, tt2 = f_cmeans(E, it, tt1, n, c, x, v, w, wt, m, d, tempv, it, objtmp)
    v = copy.deepcopy(tempv1)
    obj = objtmp


    while True:
        nbiter += 1
        kinter = kfirst
        ind = Unoccupied(c, n, d, x, v, ind)
        while True:
            for k in range(kinter):
                ceout = random.randint(0, 10000) % c
                cein = random.randint(0, 10000) % n
                ind[cein] = False
                for j in range(d):
                    tempv1[ceout][j] = x[cein][j]
            v, w, wt, tempv1, objtmp, tt2 = f_cmeans(E, it1, tt2, n, c, x, tempv1, w, wt, m, d, tempv, it, objtmp)
            tt += 1
            if objtmp < obj:
                obj = objtmp
                v = copy.deepcopy(tempv1)
                w, wt = Calculate_w(n, c, x, v, m, w, wt, d)
                kinter = kfirst
            else:
                kinter += kstep
            if kinter > kmax:
                break
        t = time.time() - stime
       # t += tt2
        if t > t0:
            break
    it1 = nbiter
    tt2 = time.time() - stime
    v, w, wt, tempv1, objtmp, tt2 = f_cmeans(E, it1, tt1, n, c, x, v, w, wt, m, d, tempv, it, objtmp)
    if objtmp < obj:
        obj = objtmp
        Newv(c, d, v, tempv1)
        v = copy.deepcopy(tempv1)
        w, wt = Calculate_w(n, c, x, v, m, w, wt, d)

    # print(np.array(w).shape)
    return v, w, tt2, obj


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Please enter the input data file... ')
        sys.exit(0)
    # Allocating memory
    inputs = sys.argv[1]
    df = pd.read_csv(inputs)
    x = df.values.tolist()
    # print(x)
    n, d = df.shape
    E = TERMINATION_THRESHOLD
    m= FUZZY_FACTOR
    objtmp = 0
    it1, obj, tt1 = 0, 0, 0
    it = 1000
    c = NUM_OF_CLUSTER
    w = np.zeros((n, c))
    tempv = np.zeros((c, d))
    wt = np.zeros((c, n))
    # ###################################START YOUR MAIN HERE################################################
    v = inite(x, n, d)
    v, w, wt, tempv, objtmp, tt2 = f_cmeans(E, it1, tt1, n, c, x, v, w, wt, m, d, tempv, it, objtmp)
    v = inite(x, n, d)
    v, w, wt, objtmp1 = fj_means(n, c, d, 1000, TERMINATION_THRESHOLD, x, v, w, wt, tempv, FUZZY_FACTOR, objtmp)
    v = inite(x, n, d)
    v, w, tt2, objtmp2 = vns1(n,d,x,v,w,wt,tempv)

    obj_str = f'\nObjective functions FCM FJM FJM-VNS = {objtmp, objtmp1, objtmp2}'
    print('*' * len(obj_str), obj_str)
    print('*' * len(obj_str))
    # #######################################################################################################
    print("below the indices value for Variable Neighborhood Search based FJM ")
    # Save the membership results
    df_result = pd.DataFrame(data=w, columns=np.arange(1, NUM_OF_CLUSTER + 1))
    df_result['label'] = df_result.idxmax(axis="columns")

    # s_score = silhouette_score(np.array(x), df_result.label, metric='euclidean')
    # d_score = davies_bouldin_score(np.array(x), df_result.label)
    # c_score = calinski_harabasz_score(np.array(x), df_result.label)
    df_result.to_csv(f'{OUTPUT_DIR}/memberships.csv', index=False)

    # Get the index
    m = FUZZY_FACTOR
    x = df.copy()
    u = df_result.copy()
    df_index = x.copy()
    df_index['labels'] = u['label']
    v = df_index.groupby('labels').mean().values.tolist()
    v = np.array(v)
    u = u.iloc[:, :-1].values.tolist()
    u = np.array(u).T
    x = np.array(x.values.tolist())
    results = []
    for method in methods:
        result = method(x, u, v, m)
        results.append(result)
    results = np.array(results)
    index_names = ['pc', 'npc', 'fhv', 'fs', 'xb', 'bh', 'bws']
    for i, r in enumerate(results):
        print(index_names[i], r)

    # Drawing
    #df_draw = df.copy()
    #df_draw['label'] = pd.to_numeric(df_result['label'])
    #plot_clusters(df_draw, 'results', 'fjmean')

    pca = PCA(2)
    draw_df = pd.DataFrame(pca.fit_transform(df), columns=['x1', 'x2'])
    draw_df['labels'] = pd.to_numeric(df_result['label'])
    colors = generate_random_color(NUM_OF_CLUSTER)
    draw_df['c'] = draw_df.labels.map(colors)

    center_pts = draw_df.groupby('labels').median().values.tolist()
    ax = plt.gca()
    for pt in center_pts:
        ax.scatter(pt[0], pt[1], marker='+', linewidths=2, s=100)
    plt.scatter(draw_df.x1, draw_df.x2, c=draw_df.c, alpha=0.6)
    plt.gca().update(dict(title='Fuzzy J-Means based VNS- PCA(2)'))
    plt.savefig(f'{OUTPUT_DIR}/clutser_plot_with_center')
    plt.show()