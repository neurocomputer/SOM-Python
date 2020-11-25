'''
Self-organizing map (SOM) Module

For fast creating a model do:

import msom
msom.fix_seed() # fix numpy random seed for experiment repeatability
model = msom.SOM() # creates a SOM model
model.fast('food')
model.BMU(7,'jet','r')
model.show_component(0,'jet','k',9)

For creating a model do:

import msom
msom.fix_seed() # fix numpy random seed for experiment repeatability
model = msom.SOM() # creates a SOM model
model.load_data('food') # specify input file name (only .csv format)
model.create(10) # creates a map with 100 units (neurons) - square 10x10
model.initw('max') # weight initialization
model.rc() # turn ON rand_control for better learning 
model.train(cnt=1000, goal=0.1, lr=0.001, pf=0, sf=0) # train your model
model.save('food') # save your model
model = msom.load('food') # load your model

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pandas
import pickle
plt.rcParams.update({'font.size': 11})
plt.rc('legend', fontsize=12)

def _save_model(model):
    '''
    Simple model save (not recommended)
    model - msom.SOM object
    '''
    fname = model.fname.split('.')[0]+'.SOM'
    with open(fname, 'wb') as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
        print('Модель сохранена как', fname)
        
def _load_model(fname):
    '''
    Simple model loading (not recommended)
    fname - model file name
    '''
    with open(fname, 'rb') as file:
        model = pickle.load(file)
    return model

def fix_seed():
    '''
    Fix numpy random seed for experiment repeatability
    '''
    np.random.seed(0)
    print('Numpy random seed is fixed!')

def euqlidean(p1, p2):
    '''
    Euclidean distance
    p1, p2 - positions of points 1 and 2
    '''
    d = np.sqrt(np.sum((p2-p1)**2))
    return d
    
def exp_f(t, a, a0):
    '''
    Monotonically decreasing learning coefficient
    a(t) = a0 * exp(-a*t)
    t - step index 
    a - speed value
    a0 - start value
    '''
    return a0 * math.exp(-a*t)
    
def influence(t, p, a, a0):
    '''
    Neighborhood function
    h(t,p) = exp(-p**2/(2*sigma(t)**2)
    sigma(t) = a0 * exp(-a*t)
    t - step index
    p - Euclidean distance between two neurons
    a - speed value
    a0 - start value
    '''
    return math.exp((-p**2)/(2*(exp_f(t, a, a0))**2))

def load(fname):
    '''
    Correct model loading (recommended)
    fname - model file name
    '''
    #for item in dir(model): print('    model.'+item+' = model_data[\''+item+'\']')
    with open(fname+'.SOM', 'rb') as file:
        model_data = pickle.load(file)
    model = SOM()
    model.N = model_data['N']
    model.all_examples = model_data['all_examples']
    model.fname = model_data['fname']
    model.goal = model_data['goal']
    model.inp_data = model_data['inp_data']
    model.inp_labels = model_data['inp_labels']
    model.learned_flag = model_data['learned_flag']
    model.lr = model_data['lr']
    model.ni = model_data['ni']
    model.performance = model_data['performance']
    model.rand_control = model_data['rand_control']
    model.training_curve = model_data['training_curve']
    model.create(model_data['nodes'])
    model.initw()
    for i in range(model.nodes):
        for j in range(model.nodes):
            model.map[i][j].weights = model_data['map'][(i,j)]
    print('SOM weights recovered from PCs memory')
    return model 

class Node():
    '''
    SOM neuron
    x,y - position on the map
    weights - weight vector of the node
    '''
    def __init__(self,x,y,size,mwv):
        '''
        Node creation
        x - x-position
        y - y-position
        mwv - maximum weight value for each input for weights initialization
        '''
        self.x = x
        self.y = y
        self.weights = np.random.randint(0, mwv, (size,)).astype(np.float64)
    
    def show(self):
        '''
        Print nodes' parameters
        '''
        print('Node {},{}. Weight vector: {}'.format(self.x,self.y,list(np.round(self.weights,3))))
        
class SOM():
    '''
    Карта Кохонена.
    Атрибуты:
    ni - количество входов каждого узла
    nodes - количество узлов по одной стороне 
    map - карта Кохонена (list of lists of Node)
    mwv - максимальное значение веса на каждом входе при инициализации весов

    '''
    fname = None #имя файла с входными данными
    inp_data = None #входные данные
    inp_labels = None #метки входных данных, при наличии
    N = None #число семплов
    ni = None #число признаков семпла
    rand_control = 'no' #контроль последовательности выбора входных семплов
    pn = None #предыдущий семпл
    learned_flag = 0 #созданная SOM еще не обучена
    
    def rc(self):
        if self.rand_control == 'no':
            self.rand_control = 'yes'
            print('rand control is ON')
        elif self.rand_control == 'yes':
            self.rand_control = 'no'
            print('rand control is OFF')
    
    def create(self, nodes):
        '''
        Создание карты.
        Создаем квадратную решетку с длиной стороны nodes.
        Пока узлов в ней нет.
        '''
        try:
            assert type(self.ni) != None
            self.nodes = nodes
            self.map = [[0 for i in range(nodes)] for i in range(nodes)]
            print('Создана решетка {} х {}'.format(self.nodes,self.nodes))
        except AssertionError:
            print('Укажите данные!')
    
    def fast(self,fname):
        '''
        Create fast your own SOM
        fname - csv input data file name
        '''
        self.load_data(fname)
        self.create(15)
        self.initw('max')
        self.rc()
        self.train(cnt=1000, goal=0.1, lr=0.001, pf=0, sf=1)
        self.save(fname)
    
    def initw(self, mwv=255):    
        '''
        Каждый узел - экземпляр класса Node.
        При инициализации веса задаются случайными значениями.
        mwv - максимальное значение веса при инициализации
        '''
        if mwv == 'max':
            mwv = self.inp_data.max()
        
        for i in range(self.nodes):
            for j in range(self.nodes):
                self.map[i][j] = Node(i,j,self.ni,mwv)
        print('Количество входов узла {}'.format(self.ni))
        print('Веса инициализированы случайными числами от 0 до {}'.format(mwv))
    
    def iteration(self, t, pf=0):
        '''
        Итерация обучения.
        W(t) = W(t-1) + a(t)h(t)*(Х-W(t-1))
        Х - входной вектор
        W(t-1) - вес нейрона на предыдущем шаге
        W(t) - новый вес
        h(t,p) - функция близости в топологии карты Кохонена
            h(t,p) = exp(-p**2/(2*sigma(t)**2)
            sigma(t) = sigma_0 * exp(-c*t)
        a(t) - коэффициент скорости обучения
            a(t) = a_0 * exp(-b*t)
        '''
        
        #1.выбираем случайный входной вектор
        n = np.random.randint(len(self.inp_data))
        #если включен контроль последовательности выбора входных семплов,
        #то следующий семпл не может быть тем же, что предыдущий
        if self.rand_control == 'yes':
            if self.pn == n:
                while True:
                    n = np.random.randint(len(self.inp_data))
                    if self.pn != n:
                        break
            self.pn = n
        self.all_examples.append(n) #сохраняем номера семплов для анализа
        if pf==1: print('Номер входного вектора {}'.format(n))
        v = self.inp_data[n]
        if pf==1: print(v)
        #2.определяем ближайший
        self.find_winner(v,pf=0)
        point_BMU = np.array(self.max_d[1]) # координаты BMU на карте
        #print(self.map[self.max_d[1][0]][self.max_d[1][1]].weights)
        #3.изменение векторов весов
        eta = exp_f(t,self.lr,1)
        for i in range(self.nodes):
            for j in range(self.nodes):
                point_2 = np.array([self.map[i][j].x,self.map[i][j].y]) # координаты узла на карте
                
                dist = euqlidean(point_BMU, point_2)
                h = influence(t,dist,0.01,10)
                delta = v - self.map[i][j].weights
                #print(eta)
                #if dist <= 5:
                self.map[i][j].weights += (eta * h * delta)
    
    def hist_examples(self):
        '''
        Гистограммма распределения примеров из обучающей выборки
        '''
        plt.hist(self.all_examples, bins=len(self.inp_data))
        plt.show()
        
    def train(self,cnt,goal,lr=0.01,pf=0,sf=0):
        '''
        learned_flag - флаг проверки, было ли обучение
        inp_data - входные данные inp_data.shape = (N, M) N семплов по M признаков
        cnt - количество итераций обучения
        goal - целевое значение показателя точности
        all_examples - все номера входных векторов при обучении
        '''
        if self.learned_flag == 1:
            print('SOM уже обучена')
            return
        assert type(self.inp_data) != None
        self.lr = lr
        self.training_curve = []
        self.all_examples = []
        self.goal = goal
        if sf==1: self.tc_fig, self.tc_ax = plt.subplots()
        for i in range(cnt):
            if pf==1: print('Итерация',i+1)
            self.iteration(i+1,pf=0)
            # under development (the 2-d mode of visualization)
            if sf==2:    
                self._show_weight_map_rgb(online='yes')
                plt.imshow(self.img)
                plt.text(self.max_d[1][1], self.max_d[1][0], '✖', ha="center", va="center", color="w")
                plt.title('Итерация '+str(i+1))
                plt.tight_layout()
                #plt.savefig('fig'+'{:4d}'.format(i+1).replace(' ', '0'))
                plt.pause(0.001)
                plt.clf()
            # ------------
            self.get_perf(pf=pf,sf=sf)
            if self.performance <= self.goal:
                break
        print('Обучение закончено за {} итераций! Точность {:.2f}'.format(i, self.performance))
        if sf!=0: plt.show()
        self.learned_flag = 1
        
    def load_data(self,fname):
        '''
        Загрузка данных из csv файла
        '''
        try:
            data = pandas.read_csv(fname+'.csv')
            print(data.head)
            if 'Label' in data.columns.tolist():
                inp_data = data.drop(['Label'], axis=1).to_numpy()
                inp_labels = data['Label'].to_list()
            else:
                inp_data = data.to_numpy()
                inp_labels = None
            assert inp_data.dtype != 'O'
        except Exception:
            print('Данные представлены в нечитаемом виде!')
            return
            
        self.fname = fname
        self.inp_data = inp_data
        self.inp_labels = inp_labels
        self.N = self.inp_data.shape[0]
        self.ni = self.inp_data.shape[1]
        print('Загружено {} семплов с {} признаками'.format(self.N,self.ni))
        
        return
    
    def BMU(self, P, cmap='binary', tc='r', pf=1, sf=1):
        '''
        Find BMU for sample in input data on position P
        '''
        inp_datum = self.inp_data[P]
        self.find_winner(inp_datum, pf=pf)
        if sf == 1:
            self.show_distances(('BMU',P), cmap=cmap, tc=tc)
            
    def find_winner(self,v,pf=0):
        '''
        Поиск BMU. Может использоваться и в процессе обучения
        и для работы
        '''
        self.d = np.zeros((self.nodes,self.nodes)) #растояние между входным и прототип векторами
        self.max_d = [sys.maxsize,[0,0]] #значение, координата
        for i in range(self.nodes):
            for j in range(self.nodes):
                self.d[i][j] = euqlidean(v,self.map[i][j].weights)
                if self.d[i][j] < self.max_d[0]:
                    self.max_d[0] = self.d[i][j]
                    self.max_d[1] = [i,j]
                elif self.d[i][j] == self.max_d[0]:
                    print('в процессе обучения найдено 2 равных узла')
                    # Если находится несколько узлов, удовлетворяющих условию, BMU выбирается случайным образом среди них.
        if pf==1: print('Distance {:.2f} to unit {},{}'.format(self.max_d[0],self.max_d[1][0],self.max_d[1][1]))
        
    def get_perf(self,pf=0,sf=0):
        '''
        Оценить ошибку карты.
        Mean distance to closest unit
        performance - текущее значение ошибки
        training_curve - кривая обучения
        '''
        all_d = []
        for item in self.inp_data:
            self.find_winner(item,pf=0)
            all_d.append(self.max_d[0])
        self.performance = np.mean(all_d)
        self.training_curve.append(float(self.performance))
        if pf==1: print('Точность {:.2f}'.format(self.performance))
        if sf==1:
            self.plot_training_curve()
    
    def plot_training_curve(self):
        '''
        Plot training curve. Can be used separetly
        or during the training
        '''
            
        plt.plot([i+1 for i in range(len(self.training_curve))], self.training_curve)
        plt.grid(linestyle='--')
        plt.ylabel('Mean distance to closest unit')
        plt.xlabel('Iteration')
        plt.axhline(y=self.goal, color='r')
        plt.title('Performance')
        plt.tight_layout()
        plt.pause(0.0001)
        plt.clf()
            
    def save(self,fname):
        '''
        Сохранение модели в память ПК
        '''
        # for item in dir(model): print('model[\''+item+'\'] = self.'+item)
        model = {}
        model['N'] = self.N # <class 'int'>
        model['all_examples'] = self.all_examples # <class 'list'>
        model['fname'] = self.fname # <class 'str'>
        model['goal'] = self.goal # <class 'float'>
        model['inp_data'] = self.inp_data # <class 'numpy.ndarray'>
        model['inp_labels'] = self.inp_labels # <class 'list'>
        model['learned_flag'] = self.learned_flag # <class 'int'>
        model['lr'] = self.lr # <class 'float'>
        model['ni'] = self.ni # <class 'int'>
        model['nodes'] = self.nodes # <class 'int'>
        model['performance'] = self.performance # <class 'float'>
        model['rand_control'] = self.rand_control # <class 'str'>
        model['training_curve'] = self.training_curve # <class 'list'>
        model['map'] = {}
        for i in range(self.nodes):
            for j in range(self.nodes):
                model['map'][(i,j)] = self.map[i][j].weights
        with open(fname+'.SOM', 'wb') as file:
            pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
            print('Модель сохранена как', fname+'.SOM')
    
    def show_distances(self,P=None,cmap='binary',tc='r',fs=10,cb='no'):
        '''
        Show distances for input datum on the map
        P - label position (optional)
        tc - text color (e.g. 'r', 'g')
        cmap - map color (e.g. 'binary', 'jet')
        '''
        plt.imshow(self.d, cmap=cmap)
        if type(P) != None:
            if P[0] == 'show_component':
                plt.title('SOM for component {}'.format(P[1]))
                if type(self.inp_labels) != None:
                    for i,item in enumerate(self.inp_labels):
                        self.BMU(i, pf=0, sf=0)
                        plt.text(self.max_d[1][1], self.max_d[1][0], item, rotation=15, ha="center", va="center", color=tc, fontsize=fs)
            elif P[0] == 'BMU':
                plt.title('Euclidean distances')
                if type(self.inp_labels) != None:
                    lb = self.inp_labels[P[1]]
                else:
                    lb = '✖'
                plt.text(self.max_d[1][1], self.max_d[1][0], lb, ha="center", va="center", color=tc)
        ax = plt.gca()
        ax.set_yticks([i for i in range(self.nodes)])
        ax.set_xticks([i for i in range(self.nodes)])
        ax.set_xticklabels(tuple([i for i in range(self.nodes)]))
        ax.set_yticklabels(tuple([i for i in range(self.nodes)]))
        if cb == 'yes':   
            plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    def show_component(self, cn, cmap='binary', tc='r', fs=10, cb='no'):
        '''
        Show the map of a component
        cn - component number
        '''
        img = np.zeros((self.nodes,self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                img[i][j] = self.map[i][j].weights[cn]
        self.d = img 
        self.show_distances(('show_component',cn), cmap=cmap, tc=tc, fs=fs, cb=cb)
        
    #todo: функции ниже требуют доработки
    def find_cmw(self):
        '''
        Найти максимальный текущий вес в карте.
        '''
        self.cmw = 0
        for i in range(self.nodes):
            for j in range(self.nodes):
                cw = self.map[i][j].weights.max()
                if cw > self.cmw:
                    self.cmw = cw
        print('Текущий максимальный вес {:.2f}'.format(self.cmw))
        
    def show_weight_map_rgb(self,online='no'):
        '''
        Карта весов в RGB.
        '''
        self.find_cmw()
        self.img = np.zeros((self.nodes,self.nodes,3),dtype=np.int32)
        for i in range(self.nodes):
            for j in range(self.nodes):
                #print(self.map[i][j].weights)
                item = int((np.sum(self.map[i][j].weights / self.cmw) / self.ni)*1023) + 1
                #print(item)
                self.img[i][j] = libsom.colors_dec[item]
        if online == 'no':
            plt.imshow(self.img)   
            plt.show()
        else:
            return
    
    def _show_weight_map_rgb(self,online='no'):
        '''
        Карта весов в RGB.
        '''
        self.img = np.zeros((self.nodes,self.nodes,3),dtype=np.int32)
        for i in range(self.nodes):
            for j in range(self.nodes):
                self.img[i][j] = self.map[i][j].weights.astype(np.int32)
        if online == 'no':
            plt.imshow(self.img)
            plt.title('Инициализация карты')
            plt.tight_layout()
            #plt.savefig('fig0000')   
            plt.show()
        else:
            return 