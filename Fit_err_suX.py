#villoni.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.odr import ODR, Model, Data

print('''istruzioni:
      - per fare un fit lineare inserire come funzione "lambda x,p: p[0]*x+p[1]"
      - fit(func,x,y,sy,guess,r1,r2) #r1,r2 sono il numero di cifre decimali    
      - fit_plot(func,x,y,sy,popt,title='title',ylabel='y',xlabel='x',xticks=0,yticks=0,loc='best',save=False) 
      - chi_squared(observed, expected, uncertainty, ddof,instruction=False)\n''')

greek=[chr(code) for code in range(945,970)]
print(greek)

#funzioni
def print_result(p,c,r1=10,r2=10):
    alphabet = [chr(code) for code in range(97,123)]
    for i in range(len(p)):
        print(f'{alphabet[i]} : {round(p[i],r1)} +/- {round(np.sqrt(c[i][i]),r2)}')
    for i in range(len(p)):
        for j in range(i+1,len(p),1):
            print(f'cov {alphabet[i]},{alphabet[j]} : {round(c[i][j],r2)}')

def chi_squared(observed, expected, uncertainty, ddof,instruction=False):
    chi_squared = np.sum((observed - expected)**2 / uncertainty**2)
    reduced_chi_squared = chi_squared / (len(observed) - ddof)
    p_value = chi2.sf(chi_squared, len(observed) - ddof)
    alpha = 0.05

    print("Chi squared:", round(chi_squared,2))
    print("Reduced chi squared:", round(reduced_chi_squared,3))
    print("P value:", round(p_value,3))
    
    if (1-p_value) < alpha:
        print("Compatibilie al 95%")
    else:
        print(f"Compatibile al : {round((p_value)*100,2)}% ")

    if instruction==True:
        print("\nInstruction:")
        print("P value è l'integrale della distribuzione chi con n df da chi_squared a inf.")



def fit(func,x,y,sx,sy,guess,r1=5,r2=5):
    func1 = lambda y, x: func(x, y)
    data = Data(x, y, wd=sx, we=sy)
    # Creazione dell'oggetto Model
    model = Model(func1)
    odr = ODR(data, model, beta0=guess)
    output = odr.run()
    popt=output.beta
    cov=output.cov_beta
    print_result(popt,cov,r1,r2)
    print('\n')
    chi_squared(y, func(x,popt), sy, len(popt))

    return popt,cov
def fit_plot(func,x,y,sx,sy,popt,title='title',ylabel='y',xlabel='x',xticks=0,yticks=0,loc='best',save=False):
    x_fit = np.linspace(x.min(),x.max(),1000)
    plt.figure(figsize=(10,5))
    plt.errorbar(x,y,xerr=sx,yerr=sy,color="black",ls='', marker='.',label='dati')
    plt.plot(x_fit,func(x_fit,popt),'-',label='fit')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xticks!=0 and yticks!=0:
            plt.yticks(yticks)
            plt.xticks(xticks)    

    plt.grid()
    plt.legend(loc=loc)
    if save==True:
         plt.savefig(f'fit_{title}.png',dpi=300)
    plt.show()
