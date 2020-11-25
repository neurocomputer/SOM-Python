import msom

msom.fix_seed() # фиксируем счетчик случайных чисел
model = msom.SOM()
fname = 'food'
model.load_data(fname)
model.create(20)
model.initw('max')
model.rc()
model.train(cnt=800, goal=0.1, lr=0.001, pf=0, sf=1)
model.save(fname)
model.show_component(0,'jet','k',9)
