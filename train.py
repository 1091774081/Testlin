# -*- coding: utf-8 -*-
import os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from config import OUTPUT_DIR, TM_CONFIG
from preprocessing import vocab

def main():
    X_train = np.load(os.path.join(OUTPUT_DIR,'X_train.npy'))
    Y_train = np.load(os.path.join(OUTPUT_DIR,'Y_train.npy'))
    X_val = np.load(os.path.join(OUTPUT_DIR,'X_val.npy'))
    Y_val = np.load(os.path.join(OUTPUT_DIR,'Y_val.npy'))

    # 布尔化可视化
    feat_vals = X_train[0]
    df = pd.DataFrame({'Feature': vocab,'Value': feat_vals})
    print(df.to_string(index=False))
    plt.figure(figsize=(12,4))
    plt.bar(range(len(vocab)), feat_vals, color=['C0' if v==1 else 'lightgray' for v in feat_vals])
    plt.xticks(range(len(vocab)), vocab, rotation=90)
    plt.yticks([0,1])
    plt.tight_layout(); plt.show()

    X_train, Y_train = X_train[:20000], Y_train[:20000]
    tm = MultiClassTsetlinMachine(TM_CONFIG['clauses'],TM_CONFIG['T'],TM_CONFIG['s'])
    history=[]
    for e in range(TM_CONFIG['epochs']):
        tm.fit(X_train,Y_train,epochs=1)
        acc=100*(tm.predict(X_val)==Y_val).mean()
        history.append(acc); print(f"Epoch {e+1} Acc {acc:.2f}%")
    plt.plot(history); plt.show()

if __name__=='__main__': main()