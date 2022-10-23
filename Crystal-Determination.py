import numpy as np
from numpy import sin, radians, sqrt
import xlsxwriter
import xlrd
import itertools
from itertools import combinations_with_replacement,combinations, permutations
from statistics import mode
import matplotlib.pyplot as plt

lamda=0.154056

face = [3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32]
diamond1 = [3, 8, 11, 16, 19, 24, 27, 31, 34, 39, 42]

"=================="
"Writing Excel File"
"=================="

fname = ['cubic struture', 'hexagonal structure']
sname = [['emperical method (indexing)', ['Common Qoutient A','analytical method (indexing)']],
         ['emperical method (indexing)', ['A value', 'C value', 'analytical method indexing']]]

print('==========================================')
print('please enter file name \neg. \n "0" for cubic structure \nor \n "1" for hexagonal structure')
fn1 = int(input('Here :', ))


fn = fname[fn1]
workbook = xlsxwriter.Workbook( fn +'.xlsx')

print('==========================================')
print('please select \n "0" choose Emperical method \nor \n select "1" Analytical Method')
EA = int(input('Here :',))
print('==========================================')

wb = xlrd.open_workbook('database.xls')
if (fn1 == 0) and (EA == 0 or EA == 1):
    I = 0
    sheet = wb.sheet_by_index(I)
elif (fn1 == 1) and (EA == 0 or EA == 1):
    I = 1
    sheet = wb.sheet_by_index(I)
        

"==========================="
"Retrieve data from database"
"==========================="

x = []
Ename = []
for col in range(1, sheet.ncols):
    Ename.append(sheet.cell_value(0,col))


print(Ename)

print("""Is there any desired element given above\n -> press 0 if 'no' \n -> press 1 if 'yes' """)
inp1 = int(input('press 0 or 1 :', ))
print('==========================================')
if inp1 == 1:
    for i in range(len(Ename)):
        print('Enter ' + str(i+1) + ' for ' + Ename[i])
    theta = int(input('Here :', ))
    print('==========================================')
    print('Result:')
    El = Ename[theta-1]
    for row in range(1,sheet.nrows):
        if fn1 == int(sheet.cell_value(row,0)):
            x.append(sheet.cell_value(row,theta))
    while x[-1] == '':
        x.remove('')

else:
    print('enter atleast 8 "2\u03B8" values')
    x = []
    for i in range(8):
        y1 = float(input(str(i+1)+' value of "2\u03B8": ', ))
        x.append(y1)
    El = input('enter element name here :', )

if len(x) > 9:
    x1 = []
    for i in range(9):
        x1.append(x[i])
    x = x1
    
premitive = np.arange(1, len(x)+1)
if len(x)>6:
    premitive = np.delete(premitive, 6)
body_center = np.arange(2, 2*len(x)+1, 2)
face_center = []
diamond = []
for i in range(len(x)):    
    face_center.append((face[i]))
    diamond.append(diamond1[i])    
    
if fn1 == 0:
    
    "========="
    "Module 01"   
    "========="
    
    if EA == 0: # E = Emperical , A = Analytical
        "======================"
        "Module 1-a (Emperical)"
        "======================"
        
        wsheet = workbook.add_worksheet(sname[fn1][EA])
        
        sin2x = np.round((sin(radians(np.array(x)/2)))**2,4)
        sinmx = np.round(sin2x/(sin(radians(min(x)/2)))**2,4)
        sin3mx = np.round(3*sinmx,4)
        h2k2l2=np.round(sin3mx,0)
    
        comb = combinations_with_replacement(np.linspace(0,9,10), 3)
    
        narray=[]
        for i in comb:
            narray.append(np.array(i))
        narray=np.array(narray)
        norm_hkl=[]
        hkl=[]
        for k in range(len(h2k2l2)):
            for j in range(len(narray)):
                    if h2k2l2[k]==np.sum(narray[j]**2):
                        hkl.append(list(reversed(narray[j])))
                        norm_hkl.append(np.sum(narray[j]**2))
    
        a = np.round((lamda/2)*np.sqrt(h2k2l2/sin2x),5)
    
        label_1a = ['2x', 'sin^{2}x', 'sin^{2}x/sin^{2}x_{min}', 'sin^{2}x/sin^{2}x_{min} * 3', 'h^{2}+k^{2}+l^{2}', 'hkl', 'a (nm)' ]
        All_1a = [x, sin2x, sinmx, sin3mx, norm_hkl, hkl, a]
        
        for i in range(len(All_1a)):
            wsheet.write(0, i,label_1a[i])
            for j in range(len(All_1a[0])):
                wsheet.write(1+j, i, str(All_1a[i][j]))
    
        HKL = []
        for check in range(len(x)):
            HKL.append(int(norm_hkl[check]))
        # if HKL == premitive:
        #     print ('cubic structure is premitive')
        # elif HKL == body_center:
        #     print('cubic structure is body centered')
        if HKL == face_center:
            print(El + ' is face centered')
        else:
            print (El + ' is not a face-centered')
            
        cos2 = (np.cos(radians(np.array(x)/2)))**2
        sin1 = sin(radians(np.array(x)/2))
        r = cos2/sin1
        
        m, b = np.polyfit(r,a,1)
        x1 = np.linspace(-0.5, max(r)+0.5)
        y1 = m*x1+b
        
        plt.plot(x1, y1)
        plt.xlabel('$cos^2\u03B8/sin\u03B8$', color = 'red')
        plt.ylabel('lattice parameter "a"', color = 'red')
        plt.title('Extrapolation "a = '+ str(np.round(b,5)) + '" against "$cos^2\u03B8/sin\u03B8$" = 0', color = 'blue')
        plt.scatter(r,a, color = 'red')
        plt.scatter(0, b, color = '#88c999')
        plt.show()
        
                    
            
    else:
        "============================"
        "Module 1-b Analytical Method"
        "============================"
        
        wsheet = []
        for i in range(len(sname[fn1][1])):
            wsheet.append(workbook.add_worksheet(sname[fn1][1][i]))
            
        sin2x = list((sin(radians(np.array(x)/2)))**2)
        
        sin2x_2, sin2x_3, sin2x_4 ,sin2x_5 ,sin2x_6, sin2x_8 = [], [], [], [], [], []
        for i in range(len(sin2x)):
            sin2x_2.append(round(sin2x[i]/2,3))
            sin2x_3.append(round(sin2x[i]/3,3))
            sin2x_4.append(round(sin2x[i]/4,3))
            sin2x_5.append(round(sin2x[i]/5,3))
            sin2x_6.append(round(sin2x[i]/6,3))
            sin2x_8.append(round(sin2x[i]/8,3))
            
        list1=[sin2x, sin2x_2, sin2x_3, sin2x_4, sin2x_5, sin2x_6, sin2x_8]
        A1 = []
        k=0
        for i in range(len(list1)):
            for j in range(1,len(list1)):
                if (list1[i][k]==list1[j][k+1]):
                    # if (set(list1[i]).intersection(list1[j])):
                    A1.append(list1[i][k])
                    k=k+1
        
        A = min(A1)
        
        a = lamda/(2*sqrt(A))
        
        h2k2l2=[]
        sinx_A = []
        for i in range(len(sin2x)):
            sinx_A.append(np.round(sin2x[i]/A,4))
            h2k2l2.append(round(sin2x[i]/A))
            
        comb = combinations_with_replacement(np.linspace(0,9,10), 3)
        
        narray=[]
        for i in comb:
            narray.append(np.array(i))
        narray=np.array(narray)
        
        hkl=[]
        for k in range(len(h2k2l2)):
            for j in range(len(narray)):
                    if h2k2l2[k]==np.sum(narray[j]**2):
                        hkl.append(list(reversed(narray[j])))
        
        label_X = ['2x', 'sin^{2}x', 'sin^{2}/2', 'sin^{2}/3', 'sin^{2}/4', 'sin^{2}/5', 'sin^{2}/6', 'sin^{2}/8']
        sinX = [x, sin2x, sin2x_2, sin2x_3, sin2x_4, sin2x_5, sin2x_6, sin2x_8] 
        
        for i in range(len(label_X)):
            wsheet[0].write(0,i,label_X[i])
            for j in range(len(sinX[0])):
                wsheet[0].write(1+j, i, sinX[i][j])
        
        label_A = ['2x', 'sin^{2}x', 'sin^{2}x/A', 'h^2 + k^2 + l^2', 'hkl']
        sinA = [x, np.array(sin2x), np.array(sinx_A), np.array(h2k2l2), np.array(hkl)]
        
        for i in range(len(label_A)):
            wsheet[1].write(0, i, label_A[i])
            for j in range(len(sinA[0])):
                wsheet[1].write(1+j, i , str(sinA[i][j]))
                
        HKL = []
        for check in range(len(body_center)):
            HKL.append(h2k2l2[check])
        HKL1 = []
        HKL2 = []
        for i in range(len(face_center)):
            HKL1.append(h2k2l2[i])
        for j in range(len(diamond)):
            HKL2.append(h2k2l2[j])
            
        if HKL == premitive.tolist():
            print (El +' crystalizes in a premitive cubic unit cell')
        if HKL == body_center.tolist():
            print (El +' crystalizes in a body centered cubic unit cell')
        if HKL1 == face_center:
            print (El +' crystalizes in a face centered cubic unit cell')
        if HKL2 == diamond:
            print(El +' crystalizes in a diamond structure')
if fn1==1:
    HKL = [[0, 0, 2],[1, 0, 0],[1, 0, 1],[1, 0, 2],[1, 0, 3],[1, 1, 0],[0, 0, 4],[1, 1, 2],[2, 0, 0]]

    if EA == 0:
        wsheet = workbook.add_worksheet(sname[fn1][EA])
        c_a =1.8563
        # wsheet = workbook.add_worksheet(sname[fn1][EA])
        value = []
        hkl = []
        for i in range(7):
            for j in range(7):
                for k in range(10):
                    if ((i+2*j) %3 !=0 or k %2==0) and (i>=j) and (i!=0 or j!=0 or k!=0):
                        hkl.append([i,j,k])
                        value.append(round((4/3)*(i**2+i*j+j**2)+k**2/c_a**2,4))
                        
        v_hkl= sorted(zip(value, hkl))
        
        c = []
        a = []
        sin2_x = []
        hkl = []
        for i in range(0, len(x)):
            hkl.append(v_hkl[i][1])
            value.append(v_hkl[i][0])
            if (v_hkl[i][1][0]==0):
                c.append(round((lamda*v_hkl[i][1][2])/(2*np.sin(np.radians(x[i]/2))),4))
            else:
                c.append(' ')
            if (v_hkl[i][1][1]>=0 and v_hkl[i][1][2]==0):
                a.append(round((lamda*np.sqrt((v_hkl[i][1][0])**2+(v_hkl[i][1][0])*(v_hkl[i][1][1])+(v_hkl[i][1][1])**2))/(np.sqrt(3)*np.sin(np.radians(x[i])/2)),4))
            else:
                a.append(' ')
            sin2_x.append((np.sin(np.radians(x[i]/2)))**2)
            
        label1 = ['2x', 'sin^{2}x', '4/3 (h^2 +h*k + k^2) + l^2/(c/a)^2', 'hkl', 'a', 'c']
        Sin1 = [x, sin2_x, value, hkl, a, c]
        
        for i in range(len(label1)):
            wsheet.write(0, i , label1[i])
            for j in range(len(Sin1[0])):
                wsheet.write(1+j, i, str(Sin1[i][j]))
        a1 = []
        x2 = []
        for i in range(len(a)):
            if a[i] != ' ':
                a1.append(a[i])
                x2.append(x[i])
        cos2 = (np.cos(radians(np.array(x2)/2)))**2
        sin1 = sin(radians(np.array(x2)/2))
        r = cos2/sin1
        
        m, b = np.polyfit(r,a1,1)
        x1 = np.linspace(-0.5, max(r)+0.5)
        y1 = m*x1+b
        
        plt.plot(x1, y1)
        plt.xlabel('$cos^2\u03B8/sin\u03B8$')
        plt.ylabel('lattice parameter "a"')
        plt.title('Extrapolated a = '+str(b), loc = 'center', fontsize = 10, color = 'blue' )
        plt.scatter(r,a1, color = 'red')
        plt.scatter(0, b, color = '#88c999')
        plt.show()
    # if EA == 1:
    #     wsheet = []
    #     for i in range(len(sname[fn1][1])):
    #         wsheet.append(workbook.add_worksheet(sname[fn1][1][i]))
    
    if EA == 1:
        
        wsheet = []
        for i in range(len(sname[fn1][1])):
            wsheet.append(workbook.add_worksheet(sname[fn1][1][i]))
            
        c_a =1.8563
        lamda = 0.154056
        sin2x = np.round(np.sin(np.radians(np.array(x)/2))**2,4)
        
        hk = []
        value = []
        for i in range(4):
            for j in range(4):
                if (i>=j and i!=0) or (i==0 and j==0):
                    hk.append([i,j])
                    value.append(i**2+i*j+j**2)
        v_hk = sorted(zip(value,hk))
        
        value = []
        hkl = []
        for i in range(7):
            for j in range(7):
                for k in range(10):
                    if ((i+2*j) %3 !=0 or k %2==0) and (i>=j) and (i!=0 or j!=0 or k!=0):
                        hkl.append([i,j,k])
                        value.append(round((4/3)*(i**2+i*j+j**2)+k**2/c_a**2,4))
                        
        v_hkl = sorted(zip(value, hkl))
        
        sin2x_3 = np.round(sin2x/v_hk[2][0],4)
        sin2x_4 = np.round(sin2x/v_hk[3][0],4)
        sin2x_7 = np.round(sin2x/v_hk[4][0],4)
        sin2x_9 = np.round(sin2x/v_hk[5][0],4)
        
        sinx = np.array([sin2x, sin2x_3, sin2x_4, sin2x_7, sin2x_9])
        
        tempold = 0
        AB = []
        for i in range(len(sinx)):
            for j in range(len(sin2x)):
                for k in range(len(sin2x)):    
                    if (abs(np.round(sinx[i][j]-sinx[i-1][k],4))<=0.0001):
                        temp = sinx[i][j]
                        if (tempold - temp)<=0.01:
                            AB.append(temp)
                        tempold = temp
        A = mode(AB)
        
        sin2x_A = sin2x - A
        sin2x_3_A = sin2x - 3*A
        
        sinA = np.array([sin2x,sin2x_A, sin2x_3_A])
        CC = []
        for i in range(len(sinA)):
            for j in range(len(x)):
                for k in range(len(x)):
                    if abs(np.round(sinA[i][j]-sinA[i-1][k],4))<=0.001:
                        CC.append(np.round(sinA[i][j],4))
        
        C = mode(CC)/4
        
        sinx_2_O = []
        AC = []
        hkl = []
        for i in range(len(x)):
            hkl.append(v_hkl[i][1])
            sinx_2_O.append(A*((v_hkl[i][1][0])**2 + v_hkl[i][1][0]*v_hkl[i][1][1] + v_hkl[i][1][1]**2) + v_hkl[i][1][2]**2*C )
            AC.append(str((v_hkl[i][1][0])**2 + v_hkl[i][1][0]*v_hkl[i][1][1] + v_hkl[i][1][1]**2)+'A'+'+'+ str(v_hkl[i][1][2]**2)+'C')
        label2 = ['2x', 'sin^{2}x', 'sin^{2}x/3', 'sin^{2}x/4', 'sin^{2]x/7', 'sin^{2}x/9', 'hkl']
        Sin2 = [x, sin2x, sin2x_3, sin2x_4, sin2x_7, sin2x_9, hkl]
        
        for i in range(len(label2)):
            wsheet[0].write(0, i, label2[i])
            for j in range(len(Sin2[0])):
                wsheet[0].write(1+j, i, str(Sin2[i][j]))
                
        label3 = ['2x', 'sin^{2}x','sin^{2}x-A', 'sin^{2}x-3A', 'hkl']
        Sin3 = [x, sin2x, sin2x_A, sin2x_3_A, hkl]
        for i in range(len(label3)):
            wsheet[1].write(0, i, label3[i])
            for j in range(len(Sin3[0])):
                wsheet[1].write(1+j, i, str(Sin3[i][j]))
                
        label4 = ['2x', 'sin^{2}x', 'A+C', 'sin^{2}x observed', 'hkl']
        Sin4 = [x, sin2x, AC, sinx_2_O, hkl]
        for i in range(len(label4)):
            wsheet[2].write(0, i, label4[i])
            for j in range(len(Sin4[0])):
                wsheet[2].write(1+j, i, str(Sin4[i][j]))


    if len(x) < 9:
        HKL1 = []
        for i in range(len(x)):
            HKL1.append(HKL[i])
        HKL = HKL1
    if HKL == hkl:
        print(El + ' crystallizes in ' + fn )
    else:
        print(El + ' does not belong to hexagonal crystal structure')
        
workbook.close()


# cos2 = (np.cos(radians(np.array(x)/2)))**2
# sin1 = sin(radians(np.array(x)/2))
# r = cos2/sin1

# plt.scatter(r,a)
# m, b = np.polyfit(r,a,1)
# x1 = np.linspace(0, max(r))
# y1 = m*x1+b
# plt.plot(x1, y1)

        

